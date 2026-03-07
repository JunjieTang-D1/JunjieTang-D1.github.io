---
layout: post
title: "Beating Claude Opus 4.5 at Kernel Generation with a 3B-Active RL Agent"
date: 2026-03-07
description: "A 30B MoE model with only 26.7M LoRA parameters generates faster NKI kernels than Claude Opus 4.5 — achieving 1.47x speedup and 94% fast rate on 250 benchmark tasks."
tags: [aws, ai, ml, agentic-ai]
toc:
  beginning: true
---

> **Disclaimer:** The views and opinions expressed in this post are my own and do not represent those of my employer.

Claude Opus 4.5 — one of the best coding models available — can't write efficient code for custom AI accelerators. We fixed that with 47 minutes of RL training and $200 of compute.

A 30B MoE model with only 26.7M LoRA parameters generates faster [NKI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/) kernels than Claude Opus 4.5 — achieving 1.47x speedup and 94% fast rate on 250 benchmark tasks. The key: reward shaping that teaches the agent to pipeline operations across multiple compute engines on [AWS Trainium](https://aws.amazon.com/ai/machine-learning/trainium/) hardware.

---

## Why kernel generation for custom AI silicon is an open problem

Most LLM-based kernel generation research targets CUDA. NVIDIA GPUs dominate training data, so frontier models already know CUDA well. But custom AI accelerators like Trainium use fundamentally different programming models with minimal representation in pretraining corpora.

[NeuronCores](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/neuron-core-v2.html) have four specialized compute engines — Tensor (matrix multiply), Vector (elementwise), Scalar (control flow), and GpSimd (transcendentals). Writing [NKI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/) kernels requires managing data across three memory levels (HBM, SBUF, PSUM), expressing computation as 2D tiles with a fixed 128-element partition dimension, and orchestrating work across all four engines simultaneously. Frontier LLMs struggle with this because they've seen very few NKI examples during pretraining.

We asked: can agentic reinforcement learning close this gap? Our approach builds on [CUDA-Agent](https://arxiv.org/abs/2602.24286) (Wu et al., 2026), which demonstrated that RL-trained agents can generate high-performance CUDA kernels, and adapts the idea to a fundamentally different hardware target.

## Teaching an agent to write hardware-optimized kernels

NKI-Agent combines data synthesis, a multi-turn tool-using agent, and hardware-aware PPO training.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/nki-agent/nki-agent-architecture.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    NKI-Agent pipeline: (1) Data Synthesis produces training tasks, (2) Agentic tool loop on <a href="https://aws.amazon.com/ec2/instance-types/trn1/">Amazon EC2 Trn1</a>, (3) RL Training with hardware-aware rewards on Amazon EC2 G5, (4) Deploy optimized kernels.
</div>

### Synthesizing 6,000 training tasks for a data-scarce domain

No NKI kernel benchmark existed, so we built one. Following the task formulation from [KernelBench](https://arxiv.org/abs/2502.10517) (Ouyang et al., 2025), we started from ~400 seed operations from PyTorch, [nki-samples](https://github.com/aws-neuron/nki-samples), and KernelBench itself, then applied combinatorial expansion across shapes, dtypes (bf16, fp32, fp8), and fusion patterns to produce ~15,000 candidates. After filtering through the [AWS Neuron](https://aws.amazon.com/machine-learning/neuron/) compiler and runtime for correctness and timing stability, we kept 6,000 curated tasks.

### Compile-verify-profile: a tight feedback loop

The agent, built on [Strands Agents](https://github.com/strands-agents) (Apache 2.0), follows a ReAct-style loop — similar to [SWE-agent](https://arxiv.org/abs/2405.15793) (Yang et al., 2024) and [Toolformer](https://arxiv.org/abs/2302.04761) (Schick et al., NeurIPS 2023) — with tools for compilation, correctness verification, performance profiling, and NKI pattern retrieval. It iterates up to 10 turns — write, compile, fix, verify, profile, optimize.

### Hardware-aware reward shaping

Prior work on RL for code generation — [CodeRL](https://arxiv.org/abs/2207.01780) (Le et al., NeurIPS 2022), [PPOCoder](https://arxiv.org/abs/2301.13816) (Shojaee et al., 2023), and [RLTF](https://arxiv.org/abs/2307.04349) (Liu et al., 2023) — uses pass/fail rewards based on compilation and test outcomes. We extend this with hardware-aware bonuses that reward efficient utilization of NeuronCore compute engines.

NeuronCores are designed for pipeline parallelism. A kernel using only the Tensor Engine leaves 75% of the chip idle. Our reward shaping explicitly encourages distributing work across engines — something Claude Opus 4.5 rarely does unprompted.

We train with [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347) (Schulman et al., 2017) using asymmetric clipping to encourage exploration of promising optimization strategies. Details of the reward formulation and training hyperparameters are the subject of ongoing work.

## Selecting the right base model with MoE

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/nki-agent/fig4_model_scaling.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    A 3B-active MoE architecture achieves the highest NKI pattern score, surpassing Claude Opus 4.5 baseline while requiring minimal training compute.
</div>

[Qwen3-Coder-30B-A3B](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) is ideal: large total parameter count provides broad code understanding, while low active count (3B) enables fast inference at 7.1 tok/s. LoRA training touches only 26.7M parameters and completes in 47 minutes. The practical takeaway: **you don't need massive compute to beat frontier models on a specialized domain — you need the right reward signal.**

## Benchmark results on 250 NKI kernel tasks

We evaluate on **NKIBench** — 250 tasks at three levels: single operations (100), fused operations (100), and full model components (50). We selected three frontier code-generation models as baselines based on their strong performance on SWE-bench and HumanEval: Claude Opus 4.5 (Anthropic), Kimi K2.5 (Moonshot AI), and GLM-4.7 (Zhipu AI). All baselines use the same agent scaffold with identical tools.

### NKI-Agent widens the gap at higher difficulty

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/nki-agent/fig1_benchmark_results.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    NKI-Agent reaches 97% at Level 1 and maintains 86% at Level 3 (full models), where Claude Opus 4.5 drops to 64%. RL training on complex fusion patterns drives this advantage.
</div>

### 1.47x geometric mean speedup over eager execution

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/nki-agent/fig5_speedup_comparison.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    NKI-Agent delivers 1.47x geometric mean speedup with 10x fewer active parameters than Claude Opus 4.5 (1.24x).
</div>

### Engine utilization rewards change agent behavior

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/nki-agent/fig2_engine_utilization.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Hardware-aware rewards produce measurable behavior change: 62.8% Tensor Engine and 51.4% Vector Engine utilization, with a 54.3% multi-engine rate — nearly double Claude Opus 4.5's 31.2%.
</div>

## Ablation: warm starting and engine bonuses matter most

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/nki-agent/fig3_ablation.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Removing warm start (RFT) and removing hardware-aware bonuses are the two largest individual impacts on agent performance.
</div>

Three findings: (1) Initial policy quality via warm starting is the largest single-component effect, confirming that stable PPO depends on a good starting point. (2) Hardware-aware bonuses are the second-largest factor, validating that general-purpose code rewards are insufficient for hardware-specific optimization. (3) Profiling feedback and domain-specific skill retrieval are complementary — one guides optimization direction, the other teaches NKI-specific patterns.

## Where Kernel Forge still struggles

At Level 3, 14% of tasks fail to produce a faster-than-eager kernel. The failure modes cluster in three areas: (1) **multi-kernel fusion** — tasks like full Transformer blocks that require coordinating data flow across multiple NKI kernels, where the agent generates correct but unoptimized glue code between kernels; (2) **complex reduction patterns** — operations like layer normalization over non-standard dimensions that require creative tile decomposition the agent hasn't learned; (3) **memory-bound workloads** — tasks where the bottleneck is HBM bandwidth rather than compute, making engine utilization rewards less informative. Addressing these likely requires richer reward signals (e.g., memory bandwidth utilization) and longer training horizons.

## Practical takeaways

**For RL-for-code researchers:** Hardware-aware reward shaping is high-leverage. Reward the *way* hardware is used, not just correctness and speed. MoE models are excellent RL bases — broad knowledge, cheap training.

**For kernel generation:** Frontier LLMs are surprisingly capable at NKI with tools (Claude Opus 4.5 gets 76% fast rate), but plateau on multi-engine optimization. A compile-verify-profile loop is essential. Warm starting matters more than any single reward component.

**For NKI developers:** Multi-engine pipelining is the key optimization axis. The 128-element partition dimension is the hardest constraint for all models.

## Get started

Replicate and extend these results using the following resources:

- [NKI programming guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/) — core documentation for writing Neuron kernels
- [nki-samples](https://github.com/aws-neuron/nki-samples) — reference kernel implementations (MIT-0)
- [Strands Agents](https://github.com/strands-agents) — open-source agent framework used in this work (Apache 2.0)
- [KernelBench](https://github.com/ScalingIntelligence/KernelBench) — the GPU kernel benchmark we adapted for NKI

For NKI kernel development, start with an [Amazon EC2 Trn1 instance](https://aws.amazon.com/ec2/instance-types/trn1/) and the [AWS Neuron SDK](https://aws.amazon.com/machine-learning/neuron/). For RL training, a single `g5.48xlarge` completes the full pipeline in under an hour for under $200.

**Open source coming soon.** We plan to release the full Kernel Forge codebase, NKIBench benchmark, and trained model weights in Q2 2026.

## References

1. **CUDA-Agent** — Wu et al., *Agentic Reinforcement Learning for CUDA Kernel Generation*, [arXiv:2602.24286](https://arxiv.org/abs/2602.24286), 2026. The direct inspiration for this work; we adapt the agentic RL approach from CUDA to NKI.
2. **KernelBench** — Ouyang et al., *Can LLMs Write GPU Kernels?*, [arXiv:2502.10517](https://arxiv.org/abs/2502.10517), 2025. Benchmark and task formulation we adapted for NKI.
3. **PPO** — Schulman et al., *Proximal Policy Optimization Algorithms*, [arXiv:1707.06347](https://arxiv.org/abs/1707.06347), 2017. Core RL algorithm; we extend with asymmetric clipping.
4. **CodeRL** — Le et al., *Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning*, NeurIPS 2022. Pioneered RL for code generation with execution feedback.
5. **PPOCoder** — Shojaee et al., *Execution-based Code Generation using Deep Reinforcement Learning*, [arXiv:2301.13816](https://arxiv.org/abs/2301.13816), 2023.
6. **RLTF** — Liu et al., *Reinforcement Learning from Unit Test Feedback*, [arXiv:2307.04349](https://arxiv.org/abs/2307.04349), 2023.
7. **SWE-agent** — Yang et al., *Agent-Computer Interfaces Enable Automated Software Engineering*, [arXiv:2405.15793](https://arxiv.org/abs/2405.15793), 2024. Influenced our tool-using agent design.
8. **Toolformer** — Schick et al., *Language Models Can Teach Themselves to Use Tools*, NeurIPS 2023.
9. **Qwen3-Coder** — Qwen Team, [Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct), 2025. Base model for Kernel Forge.
10. **Codex** — Chen et al., *Evaluating Large Language Models Trained on Code*, [arXiv:2107.03374](https://arxiv.org/abs/2107.03374), 2021.
11. **AlphaCode** — Li et al., *Competition-Level Code Generation with AlphaCode*, Science 378(6624), 2022.

---

*Patent pending. The system architecture, reward formulation, and optimization methodology described in this post are subject to intellectual property protection.*

