# NEURON Backend for ByteMLPerf Micro-Benchmark

Micro-benchmark backend for **AWS Trainium and Inferentia** accelerators using the [AWS Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/).

## Supported Hardware

- AWS Inferentia2 (inf2 instances)
- AWS Trainium (trn1/trn1n instances)
- AWS Trainium2 (trn2 instances)

## Requirements

- AWS Neuron SDK 2.x
- `torch-neuronx` >= 2.1
- `torch-xla` (matching PyTorch version)
- `neuronx-cc` (Neuron compiler)
- `aws-neuronx-runtime-lib` (Neuron runtime)

All dependencies come pre-installed on [Neuron DLAMIs](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/index.html).

## Quick Start

```bash
cd byte_micro_perf

# Single op benchmark
python launch.py --backend NEURON --device 0 --workload workloads/basic/tensor_gemm_ops/gemm.json

# All basic ops
python launch.py --backend NEURON --task all --device 0

# Specific ops
python launch.py --backend NEURON --task add,gemm,softmax --device 0

# Multiple NeuronCores (for XCCL collective ops)
python launch.py --backend NEURON --task all_reduce
```

## Op Coverage

**51 ops supported** — full parity with the GPU backend.

| Category | Ops | Provider |
|---|---|---|
| Vector Linear (4) | add, sub, mul, cast | torch |
| Vector SFU (6) | div, sin, cos, exp, log, sqrt | torch |
| Vector Reduction (4) | reduce_max, reduce_min, reduce_sum, topk | torch |
| Vector Norm (3) | layer_norm, rms_norm, softmax | torch |
| Vector Activation (2) | gelu, silu | torch |
| Vector Index (6) | embedding, gather, index_select, scatter, index_add, device2device | torch |
| Tensor GEMM (1) | gemm (float32, float16, bfloat16) | torch |
| LLM Basic (3) | scale_dynamic_quant, add_rms_norm_dynamic_quant, add_rms_norm | torch |
| LLM MOE (8) | moe_gating_gemm, moe_softmax_topk, moe_scatter_dynamic_quant, quant_matmul, moe_quant_group_gemm, moe_swiglu_dynamic_quant, swiglu_dynamic_quant, moe_gather | torch |
| LLM Attention (6) | head_rms_norm, head_rms_norm_dynamic_quant, rotary_embedding, store_kv_cache, dequant_kv_cache, flash_attention | torch / nki |
| XCCL (6) | all_reduce, reduce_scatter, all_gather, all_to_all, broadcast, p2p | torch |
| Host/Device (2) | host2device, device2host | torch |

### Flash Attention

Flash attention uses the **NKI (Neuron Kernel Interface)** `flash_fwd` kernel from `neuronxcc.nki.kernels.attention`. This is a hardware-optimized kernel that runs natively on NeuronCores.

- Provider: `nki`
- Mode: prefill only (batch_size=1, contiguous Q/K/V, causal mask)
- Dtype: bfloat16
- Decode mode with paged KV cache is not supported by the NKI kernel

### Unsupported dtypes

- `tfloat32` — NVIDIA-specific, rejected by GEMM op
- `int8` GEMM — not supported via `torch.matmul` on Neuron

## Architecture Notes

### XLA Compilation

The Neuron backend uses PyTorch/XLA. The **first run of each unique tensor shape triggers XLA compilation** through `neuronx-cc`, which can take seconds to minutes. Subsequent runs with the same shapes use cached compiled graphs from `/var/tmp/neuron-compile-cache/`.

This means:
- First benchmark run for a new workload will be slow
- Second run will be significantly faster
- The benchmark framework uses extra warmup iterations to absorb compilation overhead

### Device Management

Each NeuronCore is treated as a separate device. The backend uses `NEURON_RT_VISIBLE_CORES` environment variable to assign one NeuronCore per benchmark subprocess.

```
inf2.xlarge    → 2 NeuronCores
inf2.8xlarge   → 2 NeuronCores
inf2.24xlarge  → 12 NeuronCores
inf2.48xlarge  → 24 NeuronCores
trn1.2xlarge   → 2 NeuronCores
trn1.32xlarge  → 32 NeuronCores
trn2.48xlarge  → 64 NeuronCores
```

### Timing

No CUDA events are available. Timing uses `time.perf_counter_ns()` after explicit XLA synchronization (`xm.mark_step()` + `xm.wait_device_ops()`).

### Profiling

Kernel-level profiling (like torch.profiler with CUDA) is not currently supported. The `kernels` field in results will be empty. For Neuron-specific profiling, use [Neuron Profile](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/tools/neuron-sys-tools/neuron-profile-user-guide.html) externally.

## File Structure

```
backends/NEURON/
├── backend_neuron.py      # BackendNEURON class
├── env_neuron.py           # Default environment variables
├── provider_neuron.py      # NKI provider detection
├── README.md               # This file
└── ops/                    # 51 op mapping files
    ├── add.py
    ├── flash_attention.py  # NKI flash_fwd kernel
    ├── gemm.py             # Rejects tfloat32/int8
    └── ...
```

## Sample Results (inf2.8xlarge, NeuronCore-v2)

```
Op              dtype      Shape              Latency     Metric
gemm            bfloat16   1024×4096×4096     868 us      39.6 TFLOPS
softmax         bfloat16   1024×1024          24 us       232 GB/s
layer_norm      bfloat16   1024×1024          71 us       -
rms_norm        bfloat16   1024×1024          67 us       -
gelu            bfloat16   1024×1024          44 us       -
silu            bfloat16   1024×1024          44 us       -
topk            float32    1024×1024 k=10     31 us       -
reduce_max      float32    1024×1024          32 us       -
add             bfloat16   1024×1024          243 us      51.9 GB/s
```
