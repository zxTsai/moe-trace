# Experimental Ascend 0.23.0 hooks

CPU Python integration verified; **real A3/NPU acceptance pending**. This is not a
CUDA patch and must not be applied to vLLM 0.29.0. Supported initial model metadata:
GraniteMoeForCausalLM, non-quantized, uni/eager, synchronous, TP/PP/DP/PCP/DCP=1.

Pinned sources:

- vLLM v0.23.0: `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`
- vLLM Ascend v0.23.0: `5cb98caaadeff42b5b62b996e34bb2aaa29d20fd`

`base_files.json` records original and patched hashes of the three hook files.
`0001-trace-hooks.patch` covers **both** Python packages. It does not patch CANN,
TorchNPU, drivers, binaries or expert kernels. Routes are copied before
`fused_experts`, after weight dtype conversion. No second router calculation.

## Safe staging

Use [the CPU development guide](../../../docs/guides/ascend-cpu-development.md).
The staging tool reads original packages and creates a unique directory with
copied packages. It rejects file hash mismatches, modified/already-patched files,
missing anchors, bad Python syntax and output nested inside input packages.
The generated patch can be reproduced using the development script from clean
pinned Git commits. Do not regenerate the manifest to bypass a mismatch.

For hardware acceptance later, prepend the returned overlay directory to
`PYTHONPATH` **only in the dedicated collection process**, with moe-trace installed.
Set `MOE_TRACE_CONFIG` to an enabled collection configuration. Do not enable
`--enable-return-routed-experts` simultaneously. The NVIDIA installation wizard
does not yet install Ascend environments.

## Reversal

Stop the dedicated process, remove its overlay from `PYTHONPATH`, and restart
without `MOE_TRACE_CONFIG`. Original packages remain unchanged. Keep traces until
their validation is complete; no automatic deletion is performed.

For a disposable workspace containing both package directories, inspect a patch:

```bash
git apply --check /path/to/0001-trace-hooks.patch
git apply /path/to/0001-trace-hooks.patch
git apply --reverse --check /path/to/0001-trace-hooks.patch
git apply --reverse /path/to/0001-trace-hooks.patch
```

Never apply directly to a production installation or a dirty source checkout.
Unsupported paths include quantization, EP/EPLB, non-ALLGATHER dispatch, FlashComm,
multistream, graph replay, zero experts, simulated balancing, speculative decoding,
prefix caching, prompt embeddings, LoRA, microbatching, preemption and PD separation.
