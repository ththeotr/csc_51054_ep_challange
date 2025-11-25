import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from chiclab.nest import build_nest

from src.model import TwitterModel
from src.data import TwitterDataModule
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

import math


class TwitterTestDataset(Dataset):
    def __init__(
        self,
        filepath, 
        tab
    ):
        super().__init__()

        self.dataset = pd.read_json(filepath, lines=True)
        self.tab = tab
    
    def norm_tab(self, value, idx):
        return (value - self.tab.mean[idx]) / self.tab.std[idx]
        # return value

    def __getitem__(self, index):
        sample = self.dataset.iloc[index]

        desc = sample["user.description"]
        text = sample["text"]

        tab = torch.tensor([
            self.norm_tab(sample["user.listed_count"], 0),
            self.norm_tab(sample["user.favourites_count"], 1),
            self.norm_tab(sample["user.statuses_count"], 2),
        ], dtype=torch.float32)

        return {
            "text": text,
            "desc": desc,
            "tab": tab,
        }

    def __len__(self):
        return len(self.dataset)

class TwitterTestDataModule:
    def __init__(
        self,
        data_cfg,
        tokenizer_cfg,
        tab_cfg
    ):
        self.data_cfg = data_cfg
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg.name)
        self.tab_cfg = tab_cfg

    def collate_fn(self, batch):
        text_enc = self.tokenizer([b["text"] for b in batch], padding=True, return_tensors="pt")
        desc_enc = self.tokenizer([b["desc"] for b in batch], padding=True, return_tensors="pt")
        tabs = torch.stack([b["tab"] for b in batch])

        return {
            "text": text_enc.input_ids,
            "desc": desc_enc.input_ids,
            "text_mask": text_enc.attention_mask,
            "desc_mask": desc_enc.attention_mask,
            "tab": tabs,
        }
    
    def prepare(self):
        url = self.data_cfg.url
        self.dataset = TwitterTestDataset(
            url,
            self.tab_cfg
        )
    
    def test_dataloader(self):
        dataloader = DataLoader(
            self.dataset,
            batch_size = self.data_cfg.batch_size,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.data_cfg.num_workers
        )
        return dataloader

def plot_tsne(train_embs, train_labels, test_embs, image_path):
    tsne = TSNE(n_components=2, random_state=0)
    plot_embs = torch.cat([train_embs, test_embs])
    train_labels = torch.cat([train_labels, 1 + torch.ones(test_embs.shape[0])])
    if train_embs.shape[1] > 2:
        X_tsne = tsne.fit_transform(plot_embs)
    elif train_embs.shape[1] == 2:
        X_tsne = plot_embs

    df = pd.DataFrame({'TSNE-1': X_tsne[:, 0], 'TSNE-2': X_tsne[:, 1], 'label': train_labels})
    sns.scatterplot(data=df, x='TSNE-1', y='TSNE-2', hue='label', palette="tab10", alpha=0.7)
    # df = pd.DataFrame({'TSNE-1': Z_tsne[:, 0], 'TSNE-2': Z_tsne[:, 1]})
    # sns.scatterplot(data=df, x='TSNE-1', y='TSNE-2', palette='tab10', alpha=0.7)
    plt.title("t-SNE by label (seaborn)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title='Label')
    plt.savefig(image_path)
    plt.close()


def main(config):
    config_train = OmegaConf.load(config.config)
    model = TwitterModel(**config_train.model)
    ckpt = torch.load(config.ckpt)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(config.device)
    model.eval()

    nest = build_nest(**config_train.trainer.nest, fn=model)

    test_datamodule = TwitterTestDataModule(
        config.data,
        config_train.data.tokenizer,
        config_train.data.tab
    )
    train_data_module = TwitterDataModule(
        config_train.data.dataset,
        config_train.data.tokenizer,
        config_train.data.tab
    )

    test_datamodule.prepare()
    train_data_module.prepare()

    train_dataloader = train_data_module.train_dataloader()
    test_dataloader = test_datamodule.test_dataloader()

    acc = 0
    train_embeddings = []
    train_logits = []
    train_labels = []
    cnt = 0
    for batch in tqdm(train_dataloader):
        labels = batch["labels"]
        batch = {k: v.to(config.device) for k, v in batch.items() if k != "labels"}
        preds, embs = nest.infer(**batch)
        train_embeddings.append(embs.cpu())
        train_logits.append(preds.cpu())
        train_labels.append(labels)

        # acc += (preds.cpu() == labels).float().mean()
        # cnt += 1
        # if cnt == 10:
        #     break

    
    print(acc / len(train_dataloader))
    train_embeddings = torch.cat(train_embeddings)
    train_logits = torch.cat(train_logits)
    train_labels = torch.cat(train_labels)

    # torch.save(train_embeddings, "train_supcon_embeddings.pt")
    torch.save(train_logits, "train_supcon_logits.pt")
    # torch.save(train_labels, "train_labels.pt")

    test_embeddings = []
    test_logits = []

    cnt = 0
    for batch in tqdm(test_dataloader):
        batch = {k: v.to(config.device) for k, v in batch.items()}
        preds, embs = nest.infer(**batch)
        test_embeddings.append(embs.cpu())
        test_logits.append(preds.cpu())

        # cnt += 1
        # if cnt == 10:
        #     break


    test_embeddings = torch.cat(test_embeddings)
    test_logits = torch.cat(test_logits)

    # torch.save(test_embeddings, "test_supcon_embeddings.pt")
    torch.save(test_logits, "test_supcon_logits.pt")

    # torch.save()

    # plot_tsne(
    #     train_embeddings,
    #     train_labels,
    #     test_embeddings,
    #     "train_cluster_supcon_embs.png"
    # )
    # plot_tsne(
    #     train_logits,
    #     train_labels,
    #     test_logits,
    #     "train_cluster_supcon_logits.png"
    # )

def fix(config):
    config_train = OmegaConf.load(config.config)
    model = TwitterModel(**config_train.model)
    ckpt = torch.load(config.ckpt)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(config.device)
    model.eval()

    train_embeddings = torch.load("train_supcon_embeddings.pt")
    test_embeddings = torch.load("test_supcon_embeddings.pt")

    length = math.ceil(train_embeddings.shape[0] / 128)
    train_logits = []
    for i in tqdm(range(length)):
        batch = train_embeddings[i*128:(i+1)*128]
        with torch.no_grad():
            logits = model.fc(batch.to(config.device))
            train_logits.append(logits.cpu())
    train_logits = torch.cat(train_logits)


    length = math.ceil(test_embeddings.shape[0] / 128)
    test_logits = []
    for i in tqdm(range(length)):
        batch = test_embeddings[i*128:(i+1)*128]
        with torch.no_grad():
            logits = model.fc(batch.to(config.device))
            test_logits.append(logits.cpu())
    test_logits = torch.cat(test_logits)

    torch.save(train_logits, "train_supcon_logits_not_norm.pt")
    torch.save(test_logits, "test_supcon_logits_not_norm.pt")


if __name__ == "__main__":
    import sys
    config = OmegaConf.load(sys.argv[1])
    main(config)