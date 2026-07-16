import json
import requests
import sys


teacher = sys.argv[1]
input_file = sys.argv[2]
output_file = sys.argv[3]


urls = {
    "qwen": "http://localhost:10001/v1/chat/completions",
    "deepseek": "http://localhost:10002/v1/chat/completions",
    "gptoss": "http://localhost:10003/v1/chat/completions"
}

models = {
    "qwen": "Qwen3-8B",
    "deepseek": "DeepSeek-R1-Distill-Qwen-7B",
    "gptoss": "gpt-oss-20b"
}


url = urls[teacher]
model = models[teacher]


with open(input_file) as f:
    prompts = json.load(f)


results = []


for i, item in enumerate(prompts):

    print(f"[{i+1}/{len(prompts)}] generating...")

    r = requests.post(
        url,
        timeout=300,
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": item["prompt"]
                }
            ]
        }
    )

    if r.status_code != 200:
        print("teacher error:", r.text)
        continue

    data = r.json()

    results.append(
        {
            "instruction": item["prompt"],
            "output": data["choices"][0]["message"]["content"],
            "teacher": model
        }
    )


with open(output_file, "w") as f:
    for x in results:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")


print(f"saved {len(results)} samples to {output_file}")
