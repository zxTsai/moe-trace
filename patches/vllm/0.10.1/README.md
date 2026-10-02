# vLLM 0.10.1 integration patch

This patch is only for base commit
`aab549870df50edf0512f0a59b574f692f546465`. It adds Python-only hooks and
does not rebuild `_C` or `_moe_C`.

## Before applying

Install this repository in the same environment as editable vLLM, then verify
the exact source and preserve local work:

```bash
conda activate moe-vllm
python -m pip install -e /home/czx/code/moe-routing/code
cd /home/czx/code/vllm
git status --short
test "$(git rev-parse HEAD)" = "$(cat /home/czx/code/moe-routing/code/patches/vllm/0.10.1/base_commit.txt)"
git apply --check /home/czx/code/moe-routing/code/patches/vllm/0.10.1/0001-moe-trace-hooks.patch
git apply --check /home/czx/code/moe-routing/code/patches/vllm/0.10.1/0002-uniproc-trace-shutdown.patch
git apply --check /home/czx/code/moe-routing/code/patches/vllm/0.10.1/0003-request-lifecycle.patch
```

Do not apply if `git status` shows overlapping user changes. Apply and restart
the service:

```bash
git apply /home/czx/code/moe-routing/code/patches/vllm/0.10.1/0001-moe-trace-hooks.patch
git apply /home/czx/code/moe-routing/code/patches/vllm/0.10.1/0002-uniproc-trace-shutdown.patch
git apply /home/czx/code/moe-routing/code/patches/vllm/0.10.1/0003-request-lifecycle.patch
export MOE_TRACE_CONFIG=/home/czx/code/moe-routing/code/configs/granite-trace.yaml
# Start vLLM with --enforce-eager --no-enable-prefix-caching and no speculative model.
```

Confirm import locations before a run:

```bash
python -c 'import moe_trace, vllm; print(moe_trace.__file__); print(vllm.__file__)'
```

## Revert

First stop only the test service that you started. Check that the reverse patch
applies; this refuses to overwrite divergent edits:

```bash
cd /home/czx/code/vllm
git apply --reverse --check /home/czx/code/moe-routing/code/patches/vllm/0.10.1/0003-request-lifecycle.patch
git apply --reverse /home/czx/code/moe-routing/code/patches/vllm/0.10.1/0003-request-lifecycle.patch
git apply --reverse --check /home/czx/code/moe-routing/code/patches/vllm/0.10.1/0002-uniproc-trace-shutdown.patch
git apply --reverse /home/czx/code/moe-routing/code/patches/vllm/0.10.1/0002-uniproc-trace-shutdown.patch
git apply --reverse --check /home/czx/code/moe-routing/code/patches/vllm/0.10.1/0001-moe-trace-hooks.patch
git apply --reverse /home/czx/code/moe-routing/code/patches/vllm/0.10.1/0001-moe-trace-hooks.patch
```
