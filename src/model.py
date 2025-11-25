from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
import einops
from chiclab.arch import build_arch


class TwitterModel(nn.Module):
    def __init__(
        self,
        n_classes,
        dim,
        hf_model_name,
        text_dim,
        n_categories,
        n_numericals,
        tab_dim,
        depth,
        n_heads,
        drop_rate,
        n_spec_toks,
    ):
        super().__init__()

        self.text_encoder = AutoModel.from_pretrained(hf_model_name)
        self.desc_encoder = AutoModel.from_pretrained(hf_model_name)

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
        self.fusion_module = nn.Sequential(
            nn.Linear(text_dim * 2 + tab_dim, dim)
        )

        self.fc = nn.Sequential(
            # nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(dim, n_classes),
        )

        # self.fc = nn.Sequential(
        #     nn.GELU(),
        #     nn.Dropout(drop_rate),
        #     nn.Linear(dim, n_classes),
        # )
    
    # def _fuse_module(self, text, desc, tabl):
    #     x = torch.cat([text, desc, tabl], dim=1)
    #     x = self.fusion_module["proj"](x)
    #     x = self.fusion_module["emb"](x)
    #     # x = x + emb
    #     return x
    
    # def _tab_encode(self, tab):

    def forward(
        self,
        text, desc, tab,
        text_mask,
        desc_mask,
    ):
        text = self.text_encoder(input_ids=text, attention_mask=text_mask).last_hidden_state[:, 0]
        desc = self.desc_encoder(input_ids=desc, attention_mask=desc_mask).last_hidden_state[:, 0]
        tabl = self.tab_encoder(x_n=tab)[:, 0]

        embeddings = torch.cat([text, desc, tabl], dim=1)
        embeddings = self.fusion_module(embeddings)

        logits = self.fc(embeddings)

        # self.classifier.weight.data.copy_(F.normalize(self.classifier.weight.data, p=2, dim=1))
        # x = F.normalize(x, p=2, dim=1)

        # logits = self.classifier(x)
        # logits = x

        return logits, F.normalize(embeddings, dim=1, p=2)
        # return logits, embeddings


class TwitterNewModel(nn.Module):
    def __init__(
        self,
        n_classes,
        dim,
        hf_model_name,
        text_dim,
        n_categories,
        n_numericals,
        tab_dim,
        tab_depth,
        depth,
        n_heads,
        drop_rate,
        n_spec_toks,
    ):
        super().__init__()

        self.text_encoder = AutoModel.from_pretrained(hf_model_name)

        self.tab_encoder = build_arch(
            "ft-transformer",
            n_categories,
            n_numericals,
            tab_dim,
            tab_depth,
            n_heads,
            drop_rate,
            n_spec_toks,
        )
        self.text_decoder = build_arch(
            "ca-transformer",
            dim,
            dim,
            depth,
            n_heads,
            drop_rate,
        )
        self.tab_decoder = build_arch(
            "ca-transformer",
            dim,
            dim,
            depth,
            n_heads,
            drop_rate,
        )

        # self.fc = nn.Sequential(
        #     nn.Linear(dim, dim),
        #     # nn.GELU(),
        #     # nn.Dropout(drop_rate),
        #     # nn.Linear(dim, dim),
        # )
        self.fc = nn.Linear(dim * 2, n_classes, bias=False)

        # self.fc = nn.Sequential(
        #     nn.GELU(),
        #     nn.Dropout(drop_rate),
        #     nn.Linear(dim, n_classes),
        # )
    
    # def _fuse_module(self, text, desc, tabl):
    #     x = torch.cat([text, desc, tabl], dim=1)
    #     x = self.fusion_module["proj"](x)
    #     x = self.fusion_module["emb"](x)
    #     # x = x + emb
    #     return x
    
    # def _tab_encode(self, tab):

    def forward(
        self,
        text,
        tab,
        text_mask,
    ):
        text = self.text_encoder(input_ids=text, attention_mask=text_mask).last_hidden_state
        tab = self.tab_encoder(x_n=tab)

        text_x = self.text_decoder(x=text, ctx=tab, mask=None)[:, 0]
        tab_x = self.tab_decoder(x=tab, ctx=text, mask=None)[:, 0]

        logits = torch.cat([text_x, tab_x], dim=1)

        self.fc.weight.data.copy_(F.normalize(self.fc.weight.data, p=2, dim=1))
        logits = F.normalize(logits, p=2, dim=1)

        logits = self.fc(logits)

        return logits

class TableEmbeding(nn.Module):
    def __init__(self, n_cols, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_cols, dim))
        self.bias = nn.Parameter(torch.randn(n_cols, dim))

    def forward(self, x):
        x = einops.rearrange(x, 'b n -> b n 1')
        return x * self.weight + self.bias

class TableEncoder(nn.Module):
    def __init__(
        self,
        n_cols,
        tab_dim,
        depth
    ):
        super().__init__()
        self.n_cols = n_cols
        self.cls_token = nn.Parameter(torch.randn(1, 1, tab_dim))
        self.tab_embbed = TableEmbeding(n_cols, tab_dim)
        self.convnext = build_arch("convnext-v2-1d", dim=tab_dim, depth=depth)
    
    def forward(
        self,
        tab,
        seq_len
    ):
        tab = self.tab_embbed(tab)
        tab = F.pad(tab, (0, 0, 0, seq_len - self.n_cols - 1), value=0)
        cls_toks = einops.repeat(self.cls_token, '1 1 d -> b 1 d', b=tab.size(0))
        tab = torch.cat([cls_toks, tab], dim=1)

        tab = self.convnext(tab, None)

        return tab

class TwitterFancyModel(nn.Module):
    def __init__(
        self,
        n_classes,
        dim,
        hf_model_name,
        text_dim,
        n_numericals,
        tab_dim,
        tab_depth,
        depth,
        n_heads,
        drop_rate,
    ):
        super().__init__()

        self.text_encoder = AutoModel.from_pretrained(hf_model_name)

        # self.tab_encoder = TableEncoder(n_numericals, tab_dim, tab_depth)

        self.decoder = build_arch(
            "transformer",
            dim,
            depth,
            n_heads,
            drop_rate,
        )

        # self.fuse = nn.ModuleDict({
        #     "proj": nn.Linear(text_dim, dim),
        #     "emb": nn.Sequential(
        #         nn.Conv1d(dim, dim, 31, 1, 15, 1, 16),
        #         nn.Mish(),
        #         nn.Conv1d(dim, dim, 31, 1, 15, 1, 16),
        #         nn.Mish(),
        #     )
        # })
        self.fc = nn.Linear(dim, n_classes, bias=False)
    
    def _fuse_encode(self, text_emb, tab_emb):
        # x = self.fuse["proj"](torch.cat([text_emb, tab_emb], dim=-1))
        x = self.fuse["proj"](text_emb)

        x = x.transpose(1, 2)
        x_emb = self.fuse["emb"](x)
        x = x.transpose(1, 2)
        x_emb = x_emb.transpose(1, 2)

        return x + x_emb

    def forward(
        self,
        text,
        tab,
        text_mask,
    ):
        text_emb = self.text_encoder(input_ids=text, attention_mask=text_mask).last_hidden_state
        # tab_emb = self.tab_encoder(tab, text_emb.shape[1])

        # x = self._fuse_encode(text_emb, None)

        x = self.decoder(text_emb, mask=None)[:, 0]

        self.fc.weight.data.copy_(F.normalize(self.fc.weight.data, p=2, dim=1))
        x = F.normalize(x, p=2, dim=1)

        x = self.fc(x)

        return x

