import json
import random

with open("data/train_clean_v1.jsonl") as f:
    dataset = [json.loads(l.strip()) for l in f.readlines()]

random.shuffle(dataset)

Nk = len(dataset) // 10

for i in range(10):
    valid = dataset[i*Nk:(i+1)*Nk+1]
    train = dataset[:i*Nk] + dataset[(i+1)*Nk+1:]

    with open(f"data/cross-validation/train_clean_v1_k{i}.jsonl", "w") as f:
        for item in train:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    with open(f"data/cross-validation/valid_clean_v1_k{i}.jsonl", "w") as f:
        for item in valid:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

