import numpy as np
from tqdm import tqdm

def find_mean_std(dataset):
    tmp_ls = [[], [], []]
    for sample in tqdm(dataset):
        tmp_ls[0].append(sample["user.listed_count"])
        tmp_ls[1].append(sample["user.favourites_count"])
        tmp_ls[2].append(sample["user.statuses_count"])
    
    return np.mean(tmp_ls, axis=1), np.std(tmp_ls, axis=1)

if __name__ == "__main__":
    import json

    with open("data/test_clean_v1.jsonl") as f:
        dataset = [json.loads(l.strip()) for l in f.readlines()]
    
    with open("data/train_clean_v1.jsonl") as f:
        dataset.extend([json.loads(l.strip()) for l in f.readlines()])

    print(find_mean_std(dataset))