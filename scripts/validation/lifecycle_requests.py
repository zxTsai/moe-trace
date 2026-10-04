"""Real HTTP stop/length/disconnect lifecycle workload (local test service)."""
import argparse
import json
import time
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8000/v1/completions"
MODEL = "granite-moe"


def request(**kwargs):
    payload = dict(model=MODEL, temperature=0, **kwargs)
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=URL)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output", type=Path, default=Path("/tmp/moe-trace-lifecycle-client.json"))
    args = parser.parse_args()
    URL, MODEL = args.url, args.model
    results = []
    # Explicit token-stop is enforced inside the scheduler, unlike string-stop
    # processing in the API output processor. First generated newline is token 203.
    with request(prompt="Route this request.", max_tokens=64,
                 stop_token_ids=[203], ignore_eos=True) as response:
        out = json.load(response)
        results.append(dict(case="stop", api_id=out["id"],
                            finish_reason=out["choices"][0]["finish_reason"], usage=out["usage"]))
    with request(prompt="Explain a GPU.", max_tokens=4, ignore_eos=True) as response:
        out = json.load(response)
        results.append(dict(case="length", api_id=out["id"],
                            finish_reason=out["choices"][0]["finish_reason"], usage=out["usage"]))
    with request(prompt="Explain expert routing in detail.", max_tokens=512,
                 ignore_eos=True, stream=True) as response:
        chunks = []
        for line in response:
            if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                out = json.loads(line[6:])
                if out["choices"][0]["text"]:
                    chunks.append(out)
                if len(chunks) == 2:
                    break
        if len(chunks) != 2:
            raise RuntimeError("stream ended before cancellation could be tested")
        results.append(dict(case="cancel", api_id=chunks[-1]["id"],
                            received_chunks=len(chunks), requested_max_tokens=512))
    # Give the server's disconnect watcher time to deliver an abort.
    time.sleep(2)
    with request(prompt="Still serving?", max_tokens=2, ignore_eos=True) as response:
        out = json.load(response)
        results.append(dict(case="after_cancel", api_id=out["id"],
                            finish_reason=out["choices"][0]["finish_reason"], usage=out["usage"]))
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
