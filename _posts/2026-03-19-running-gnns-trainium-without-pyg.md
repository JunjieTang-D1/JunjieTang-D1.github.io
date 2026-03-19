---
layout: post
title: "Running Graph Neural Networks on AWS Trainium Without PyTorch Geometric"
date: 2026-03-19
description: "PyTorch Geometric doesn't compile on XLA. We built neuron-pyg — a ~1,500-line drop-in replacement — and ran VectorWorld's full 45M-parameter VAE encoder on Trainium with pretrained weights. Byte-identical to PyG outputs. Inference at 55.7ms, training loss drops 84% in 50 steps."
tags: [aws, ai, ml, trainium, gnn, xla]
toc:
  beginning: true
---

> **Disclaimer:** The views and opinions expressed in this post are my own and do not represent those of my employer.

The GNN community has a hidden dependency problem. PyTorch Geometric — the library underneath virtually every graph neural network in production, with 23,000+ GitHub stars — hardcodes CUDA scatter kernels into its critical path. If your accelerator doesn't support CUDA, your GNN doesn't run.

This isn't a minor compatibility issue. It means every GNN workload — autonomous driving scene understanding, drug discovery, recommendation systems — is locked to one hardware vendor. AWS Trainium, Intel Gaudi — all blocked. [PyG issue #1584](https://github.com/pyg-team/pytorch_geometric/issues/1584) has been open since **2020**. Five years, no fix.

We decided to fix it ourselves. [neuron-pyg](https://github.com/JunjieTang-D1/neuron-pyg) is ~1,500 lines of pure PyTorch that replaces PyG's core operations with XLA-compatible implementations. We validated it by running [VectorWorld](https://arxiv.org/abs/2603.17652)'s full 45M-parameter VAE encoder on Trainium with pretrained weights — **byte-identical outputs** to the original PyG implementation, inference at 55.7ms, and training convergence verified over 50 steps.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/neuron-pyg/fig_architecture.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    neuron-pyg replaces PyG's CUDA-specific scatter kernels with XLA-compatible pure PyTorch operations. VectorWorld's 45M-parameter VAE encoder runs on CPU and Trainium with only 2 import lines changed.
</div>

---

## Why this matters beyond Trainium

PyG's architecture assumes runtime access to CUDA kernels. This was a reasonable design choice in 2019 when PyG was created. It's now a liability — one that blocks an entire class of models from running on XLA-based accelerators like Trainium.

## The three operations that break everything

PyTorch Geometric's XLA incompatibility comes from exactly three categories of operations:

**CUDA-specific scatter kernels.** `torch_scatter` provides `scatter_add`, `scatter_mean`, `scatter_max` as custom CUDA extensions. No XLA lowering exists.

**Dynamic shapes from sparse operations.** GNN neighborhoods are variable-size — node 0 has 3 neighbors, node 1 has 50. PyG handles this with dynamic indexing that XLA's ahead-of-time compiler can't trace.

**Python control flow in critical paths.** A naive `scatter_max` iterates `for i in range(dim_size): if mask.any()...` — Python control flow that breaks XLA graph tracing.

neuron-pyg replaces each with XLA-compatible alternatives:

| Operation | PyG (CUDA) | neuron-pyg (XLA) |
|-----------|-----------|-----------------|
| `scatter_add` | Custom CUDA kernel | `Tensor.scatter_add_()` |
| `scatter_mean` | CUDA kernel + count | `scatter_add_()` / count |
| `scatter_max` | CUDA kernel | Vectorized one-hot + `amax()` |
| `softmax` | `torch_scatter` + `scatter_max` | Gradient-safe, `-1e9` clamping |
| `MessagePassing` | Dynamic dispatch + CUDA | Static signature parsing |

### The hardest operation: scatter_max

`scatter_max` is where naive XLA approaches fail. The standard algorithm — iterate over groups, apply max — uses Python control flow. We replace it with fully vectorized one-hot masking:

```python
# For each group g, compute max over src[index == g]
one_hot = (index.unsqueeze(0) == 
           torch.arange(dim_size, device=index.device).unsqueeze(1))
masked = torch.where(oh_expanded, src_expanded,
                     torch.tensor(-1e9, dtype=src.dtype))
output = masked.amax(dim=1)
```

**Why `-1e9` instead of `-inf`?** This was our most expensive debugging lesson. Inference with pretrained weights worked perfectly with `-inf`. But training produced NaN losses — the gradient of `exp(x - (-inf))` is undefined, and this pathology only manifests in the backward pass. Clamping to `-1e9` eliminates the issue while being functionally equivalent for all practical attention score ranges (no real attention logit exceeds 1e9).

## Proof: byte-identical outputs

Claims of "drop-in replacement" are easy to make and hard to verify. We ran a systematic equivalence test — loading identical weights into both the original PyG-based VectorWorld layers and their neuron-pyg replacements, feeding identical inputs, and comparing outputs element-by-element:

| Layer | Max Abs Error | Mean Abs Error | Identical? |
|-------|---------------|----------------|------------|
| ResidualMLP | 0.00e+00 | 0.00e+00 | ✓ |
| AttentionLayer (homogeneous) | 0.00e+00 | 0.00e+00 | ✓ |
| AttentionLayer (bipartite) | 0.00e+00 | 0.00e+00 | ✓ |
| EdgeFeatureUpdate | 0.00e+00 | 0.00e+00 | ✓ |
| AutoEncoderBlock (agents) | 0.00e+00 | 0.00e+00 | ✓ |
| AutoEncoderBlock (lanes) | 0.00e+00 | 0.00e+00 | ✓ |
| AutoEncoderBlock (lane_conn) | 0.00e+00 | 0.00e+00 | ✓ |

Not "close enough." Not "within floating-point tolerance." **Bit-for-bit identical.** The outputs are the same because the underlying PyTorch operations are the same — we just removed the CUDA-specific dispatch layer.

## Validation: VectorWorld's VAE encoder

[VectorWorld](https://arxiv.org/abs/2603.17652) (Jiang et al., 2026) is a streaming world model for autonomous driving that uses PyG extensively. We chose it as our validation target because its VAE encoder exercises *every* neuron-pyg primitive:

- **45M parameters** across 3 transformer blocks × 4 attention layers
- Lane-to-lane self-attention (768-dim, 8 heads)
- Agent-to-agent self-attention (384-dim, 8 heads)
- Lane-to-agent cross-attention (bipartite)
- Edge feature updates via `edge_updater()`
- Lane-to-query attention for conditional distribution
- Dual-branch agent embedding with learned gating

The migration: **2 import lines changed** in the core layers file (`layers.py`) across 23,000 lines of VectorWorld code. No model modifications. Pretrained checkpoint loads directly.

## Results

### Inference with pretrained weights on Trainium

| Metric | Value |
|--------|-------|
| Checkpoint keys matched | 424 / 424 |
| Output shapes | All correct (agent [30,18], lane [100,24], cond_dis [1,101]) |
| NaN / Inf | None |
| Agent μ range | [-19.0, 23.3] |
| **Median inference latency** | **55.7ms** |
| XLA compile (first run) | 53.9s (30+ HLO graphs) |
| XLA compile (cached) | <1ms |

### Training convergence on Trainium

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/neuron-pyg/fig_training_loss.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    VectorWorld VAE encoder training on trn1.2xlarge. Loss drops 84.5% over 50 steps (random initialization, synthetic data, AdamW lr=1e-4, kl_weight=0.006). Full gradient flow through 12 attention layers with scatter operations.
</div>

| Metric | Value |
|--------|-------|
| Loss (start → end) | 456.5 → 70.9 (−84.5%) |
| NaN / Inf | None |
| **Median step time** | **178.6ms** (fwd + bwd + optimizer) |
| XLA compile (fwd+bwd) | 313.1s |

**Caveat on training results:** These 50 steps use random initialization on synthetic data — they prove that gradients flow correctly through all 12 attention layers and the optimizer updates weights as expected. This is not a full training run to convergence on real Waymo data. That would require the full VectorWorld data pipeline, which is orthogonal to the neuron-pyg contribution.

### Realistic driving scenario validation

To go beyond random tensors, we validated the pretrained encoder on five realistic Waymo-format driving scenarios — each with proper road geometry, vehicle dynamics, and traffic topology:

| Scenario | Agents | Lanes | Agent μ Range | Latency | NaN |
|----------|--------|-------|---------------|---------|-----|
| Highway (3-lane, 80 km/h) | 20 | 60 | [-11.8, 17.1] | 84.3ms | ✓ |
| T-Intersection (vehicles + pedestrians) | 20 | 80 | [-12.2, 14.5] | 89.2ms | ✓ |
| Roundabout (vehicles + cyclists) | 12 | 50 | [-10.1, 14.3] | 72.7ms | ✓ |
| Parking lot (low-speed maneuvers) | 8 | 30 | [-11.1, 14.3] | 57.8ms | ✓ |
| Highway merge (lane change dynamics) | 25 | 70 | [-13.1, 13.3] | 84.6ms | ✓ |

Each scene uses realistic parameters: proper vehicle dimensions (car 4.5×2.0m, pedestrian 0.5×0.5m, cyclist 1.8×0.6m), plausible velocities (highway 15–25 m/s, parking 1–3 m/s), trajectory histories as motion polylines, and topologically correct lane connections (predecessor/successor/left/right).

The pretrained encoder produces **semantically meaningful latent distributions** — the model processes realistic driving scenes through neuron-pyg's operations and produces non-degenerate, structured outputs. This isn't a proof of full model correctness (that requires the Waymo evaluation pipeline), but it demonstrates that neuron-pyg preserves the pretrained model's learned representations.

**Caveat:** These scenes use synthetic geometry that mimics Waymo format — realistic shapes and dynamics, but not actual sensor data. The Waymo Open Dataset requires a license agreement. Full validation on licensed Waymo data is future work.

### Three-platform scatter performance

### XLA compilation: a one-time cost

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/neuron-pyg/fig_xla_amortization.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Left: One-time XLA compilation cost (inference 53.9s, training 313.1s). Right: Amortized cost per training step — after 1000 steps, overhead drops to 492ms/step, converging to the 178.6ms cached step time.
</div>

The Neuron compiler persists compiled NEFFs to disk. On re-runs with the same model and shapes, all graphs load from cache — reducing warmup from minutes to milliseconds. **For production: compile once, run thousands of times.**

## What didn't work (and what we learned)

**`scatter_reduce_` with `amax` on XLA.** PyTorch 1.12+ provides `scatter_reduce_()` which should handle `amax` natively. On the Neuron SDK, this either fails to compile or produces incorrect results with `include_self=True`. We fell back to the one-hot vectorized approach.

**`torch.isinf()` in the backward pass.** Our initial softmax used `torch.where(torch.isinf(max_src), ...)` to handle empty groups. This compiled for inference but produced NaN gradients during training — `isinf` creates a boolean mask whose gradient is undefined. Replacing with arithmetic clamping (`clamp(min=-1e9)`) fixed the issue.

**`torch.randint` on XLA devices.** Creating random index tensors directly on XLA produces unexpected values. Fix: create on CPU, then `.to(device)`. Known XLA limitation.

**NeuronCore state after failed runs.** If compilation fails, the NeuronCore can enter a bad state where `nrt_init()` fails on subsequent attempts. Fix: kill all Python processes using the NeuronCore and wait 3 seconds. We hit this repeatedly during development.

## Design principles

**XLA-first, not XLA-compatible.** Every operation is designed for XLA from the start — no `try: cuda_kernel(); except: fallback()`. If an operation can't be expressed as a static graph, we find an alternative algorithm.

**Gradient safety over mathematical purity.** Using `-1e9` instead of `-inf` for empty groups is technically incorrect. But `-inf` causes NaN gradients on XLA. We choose "works correctly in practice" over "mathematically precise but unusable."

**Zero PyG dependency at runtime.** neuron-pyg imports only `torch`. It works on any XLA backend — Trainium, Inferentia — without PyG's CUDA dependencies.

## Limitations

**Memory overhead.** The vectorized `scatter_max` creates a `[dim_size, N, D]` intermediate tensor. For graphs with >100K edges, this becomes significant. Future work: a custom [NKI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/) kernel for scatter operations.

**Single-device only.** Not tested with `torch_xla.distributed`. Multi-NeuronCore training is future work.

**No real Waymo data.** We validated with realistic synthetic scenes matching VectorWorld's format and with pretrained weights. Full validation on the licensed Waymo Open Dataset is future work.

## Implications

**For GNN practitioners:** If your model uses PyG and you want to run it on Trainium, `pip install neuron-pyg` and swap your PyG imports. 67 tests verify correctness. Byte-identical outputs guarantee your model behaves the same.

**For PyG maintainers:** We plan to submit a PR adding XLA-compatible scatter and softmax backends, building on [issue #1584](https://github.com/pyg-team/pytorch_geometric/issues/1584). The backends can coexist with CUDA — activated when `device.type == 'xla'`.

**For the accelerator ecosystem:** The GNN community's CUDA dependency is not a fundamental algorithmic limitation. It's three operations. ~1,500 lines fix it. Every XLA-based accelerator — Trainium, Gaudi — benefits from the same approach.

## What's next

- **PyG upstream PR**: Submit XLA-compatible backends to PyTorch Geometric
- **NKI scatter kernel**: Custom Neuron kernel for `scatter_max` to optimize memory access patterns
- **Trainium2 validation**: trn2 (667 TFLOPs/chip, 3.5× over trn1) should improve both compile and runtime
- **Full Waymo validation**: End-to-end VectorWorld training on real autonomous driving data
- **Additional model validations**: QCNet, DSVT, PointPillars

---

*All benchmarks: trn1.2xlarge (Neuron SDK 2.9, neuronx-cc 2.23). VectorWorld checkpoint: [Jck1998/vectorworld](https://huggingface.co/Jck1998/vectorworld) (Waymo VAE). Code and equivalence tests: [neuron-pyg](https://github.com/JunjieTang-D1/neuron-pyg). 67 unit tests, Apache 2.0 license.*
