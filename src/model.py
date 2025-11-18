from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel

from chiclab.arch import build_arch


class TwitterModel(nn.Module):
    def __init__(
        self,
        dim,
        text_dim,
        hf_model_name,
        n_categories: List[int],
        n_numericals: int,
        tab_dim: int,
        depth: int,
        n_heads: int,
        drop_rate: float,
        n_spec_toks: int = 2,
    ):
        super().__init__()

        self.text_encoder = AutoModel.from_pretrained(hf_model_name)

        self.tab_encoder = build_arch(
            "ft-transformer",
            n_categories,
            n_numericals,
            tab_dim,
            depth,
            n_heads,
            drop_rate,
            n_spec_toks,
        )
        self.fusion_module = nn.ModuleDict({
            "proj": nn.Linear(tab_dim + 2 * text_dim, dim),
            "emb": nn.Sequential(
                nn.Linear(dim, dim),
                nn.Mish(),
                nn.Linear(dim, dim),
                # nn.Mish()
            )
        })
        self.fc = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.Mish(),
            nn.Dropout(drop_rate),
            nn.Linear(dim * 2, dim),
            # nn.Mish(),
            # nn.Dropout(drop_rate),
        )
    
    def _fuse_module(self, text, desc, tabl):
        x = torch.cat([text, desc, tabl], dim=1)
        x = self.fusion_module["proj"](x)
        x = self.fusion_module["emb"](x)
        # x = x + emb
        return x

    def forward(
        self,
        text, desc, tab,
        text_mask,
        desc_mask,
    ):
        text = self.text_encoder(input_ids=text, attention_mask=text_mask).last_hidden_state[:, 0]
        desc = self.text_encoder(input_ids=desc, attention_mask=desc_mask).last_hidden_state[:, 0]
        tabl = self.tab_encoder(x_n=tab)

        embeddings = self._fuse_module(text, desc, tabl)

        logits = self.fc(embeddings)
        # logits = logits + embeddings

        return logits, embeddings