"""Small, reproducible mixed-phase/concurrent/chunked GPU request workload."""
import concurrent.futures
import json
import time
import urllib.request


def send(label, prompt, length, delay):
    time.sleep(delay)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    payload = json.dumps({
        "model": "/home/czx/code/moe-routing/models/granite-3.1-1b-a400m-instruct",
        "prompt": prompt, "max_tokens": length, "temperature": 0,
        "ignore_eos": True,
    }).encode()
    request = urllib.request.Request("http://127.0.0.1:18001/v1/completions",
                                     data=payload, headers={"Content-Type": "application/json"})
    with opener.open(request, timeout=180) as response:
        result = json.load(response)
    return {"client_label": label, "api_id": result["id"], "usage": result["usage"]}


if __name__ == "__main__":
    cases = [("A", "Explain GPU computation. ", 48, 0),
             ("B", "Describe routing. " * 10, 12, 1),
             ("C", "Hello world. ", 6, 2)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        outputs = list(pool.map(lambda case: send(*case), cases))
    outputs.append(send("chunked", "chunk " * 150, 2, 0))
    print(json.dumps(outputs, indent=2))
