# vLLM 0.10.1 integration patch

This patch is only for base commit
`aab549870df50edf0512f0a59b574f692f546465`. It adds Python-only hooks and
does not rebuild `_C` or `_moe_C`.

For new users, prefer the [environment wizard](../../../docs/guides/environment-setup.md)
which stages an isolated copy. The source-patch workflow below is only for a clean,
self-managed copy at the exact base commit, not an instruction to overwrite installed code.

## Before applying

Install this repository in the same environment as editable vLLM, then verify
the exact source and preserve local work:

```bash
# Activate your own compatible vLLM environment; run initially from the tool repository.
TRACE_REPO=$(pwd)
TRACE_VLLM_SOURCE=/absolute/path/to/your/vllm-source
cp configs/examples/vllm-0.10.1.yaml trace.yaml
python -m pip install --no-deps --no-build-isolation -e "$TRACE_REPO"
cd "$TRACE_VLLM_SOURCE"
git status --short
test "$(git rev-parse HEAD)" = "$(cat "$TRACE_REPO/patches/vllm/0.10.1/base_commit.txt")"
git apply --check "$TRACE_REPO/patches/vllm/0.10.1/0001-moe-trace-hooks.patch"
git apply --check "$TRACE_REPO/patches/vllm/0.10.1/0002-uniproc-trace-shutdown.patch"
git apply --check "$TRACE_REPO/patches/vllm/0.10.1/0003-request-lifecycle.patch"
```

Do not apply if `git status` shows overlapping user changes. Apply and restart
the service:

```bash
git apply "$TRACE_REPO/patches/vllm/0.10.1/0001-moe-trace-hooks.patch"
git apply "$TRACE_REPO/patches/vllm/0.10.1/0002-uniproc-trace-shutdown.patch"
git apply "$TRACE_REPO/patches/vllm/0.10.1/0003-request-lifecycle.patch"
export MOE_TRACE_CONFIG="$TRACE_REPO/trace.yaml"
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
cd "$TRACE_VLLM_SOURCE"
git apply --reverse --check "$TRACE_REPO/patches/vllm/0.10.1/0003-request-lifecycle.patch"
git apply --reverse "$TRACE_REPO/patches/vllm/0.10.1/0003-request-lifecycle.patch"
git apply --reverse --check "$TRACE_REPO/patches/vllm/0.10.1/0002-uniproc-trace-shutdown.patch"
git apply --reverse "$TRACE_REPO/patches/vllm/0.10.1/0002-uniproc-trace-shutdown.patch"
git apply --reverse --check "$TRACE_REPO/patches/vllm/0.10.1/0001-moe-trace-hooks.patch"
git apply --reverse "$TRACE_REPO/patches/vllm/0.10.1/0001-moe-trace-hooks.patch"
```
