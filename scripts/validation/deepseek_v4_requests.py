"""Send three concurrent text requests to the PD proxy; not a performance benchmark."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="PD proxy /v1/chat/completions URL")
    parser.add_argument("--model", default="dsv4")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    prompts = ["What is an expert in a mixture of experts model?",
               "Explain how a mixture of experts model routes tokens, using a short example.",
               "Explain token routing and expert load imbalance in a mixture of experts model. "
               "Discuss how different requests might choose different experts."]
    def request(i):
        body = {"model": args.model, "messages": [{"role": "user", "content": prompts[i]}],
                "max_tokens": [16, 24, 32][i], "stream": False}
        req = urllib.request.Request(args.url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as response:
                result = json.load(response)
                if not result.get("choices") or "error" in result:
                    raise ValueError("response has no completion choices")
                return {"client_request": i, "status": response.status,
                        "response_id": result.get("id"), "usage": result.get("usage"),
                        "elapsed_seconds": time.monotonic() - start}
        except (urllib.error.URLError, ValueError, OSError) as exc:
            return {"client_request": i, "error": str(exc)}
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(request, range(3)))
    print(json.dumps({"requests": results,
        "note": "Concurrent clients do not prove same-execution batching; inspect collected routes."}, indent=2))
    return 0 if all(r.get("status") == 200 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
