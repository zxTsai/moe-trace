# Development baseline (2026-10-02)

| Item | Observed |
|---|---|
| Active shell | Conda `base`, Python 3.13.12 |
| Required environment | `moe-vllm`, Python 3.11 |
| PyTorch | 2.7.1+cu128 |
| Transformers | 4.55.2 |
| xFormers | 0.0.31 |
| vLLM | 0.10.1, editable import from `/home/czx/code/vllm` |
| vLLM Git | branch `moe-trace-dev`, clean, commit `aab549870df50edf0512f0a59b574f692f546465` |
| Native extensions | `vllm._C` and `vllm._moe_C` import successfully |
| Model | local GraniteMoe, 24 layers, 32 experts/layer, Top-8 |
| GPU in this agent session | unavailable: NVML blocked and `torch.cuda.is_available()` false |
| Output root | did not exist at inspection time; tool creates it only when enabled |

The user-provided environment record says the model service has previously
returned HTTP 200 on this machine. That is historical evidence, not a test run
performed by this implementation. All GPU acceptance cases remain NOT RUN in
this session until GPU access is available.
