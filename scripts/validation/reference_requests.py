"""Small, reproducible mixed-phase/concurrent/chunked GPU request workload."""
import concurrent.futures
import argparse
import json
import time
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8000/v1/completions"
MODEL = "granite-moe"


def send(label, prompt, length, delay):
    time.sleep(delay)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt, "max_tokens": length, "temperature": 0,
        "ignore_eos": True,
    }).encode()
    request = urllib.request.Request(URL,
                                     data=payload, headers={"Content-Type": "application/json"})
    with opener.open(request, timeout=180) as response:
        result = json.load(response)
    return {"client_label": label, "api_id": result["id"], "usage": result["usage"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=URL)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    URL, MODEL = args.url, args.model
    cases = [("A", "Explain GPU computation. ", 48, 0),
             ("B", "Describe routing. " * 10, 12, 1),
             ("C", "Hello world. ", 6, 2)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        outputs = list(pool.map(lambda case: send(*case), cases))
    outputs.append(send("chunked", "chunk " * 150, 2, 0))
    result = json.dumps(outputs, indent=2)
    if args.output:
        args.output.write_text(result + "\n", encoding="utf-8")
    print(result)
