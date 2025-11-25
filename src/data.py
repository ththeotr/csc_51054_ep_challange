import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer

from chiclab.feed import ChicDataModule


class TwitterDataset(Dataset):
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
        label = sample["label"]

        text = text + " description de l'utilisateur: " + desc

        tab = torch.tensor([
            self.norm_tab(sample["user.listed_count"], 0),
            self.norm_tab(sample["user.favourites_count"], 1),
            self.norm_tab(sample["user.statuses_count"], 2),
            # sample["user.listed_count"],
            # sample["user.favourites_count"],
            # sample["user.statuses_count"],
        ], dtype=torch.float32)

        return {
            "text": text,
            "tab": tab,
            "label": label
        }

    def __len__(self):
        return len(self.dataset)


class TwitterDataModule(ChicDataModule):
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
        # desc_enc = self.tokenizer([b["desc"] for b in batch], padding=True, return_tensors="pt")
        tabs = torch.stack([b["tab"] for b in batch])
        labels = torch.LongTensor([b["label"] for b in batch])

        return {
            "text": text_enc.input_ids,
            # "desc": desc_enc.input_ids,
            "text_mask": text_enc.attention_mask,
            # "desc_mask": desc_enc.attention_mask,
            "tab": tabs,
            "labels": labels
        }
    
    def prepare(self):
        url = self.data_cfg.url
        self.train_dataset = TwitterDataset(
            url.train,
            self.tab_cfg
        )
        self.valid_dataset = TwitterDataset(
            url.valid,
            self.tab_cfg
        )
    
    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size = self.data_cfg.batch_size,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.data_cfg.num_workers
        )
        return dataloader
    
    def valid_dataloader(self):
        dataloader = DataLoader(
            self.valid_dataset,
            batch_size = self.data_cfg.batch_size,
            pin_memory=True,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.data_cfg.num_workers
        )
        return dataloader