import re
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel



# ----------------- 文本编码模型 -----------------

class TextEncoder(nn.Module):
    def __init__(self, pretrained_dir, embed_dim=256, dropout=0.1):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_dir, local_files_only=True)
        self.bert = BertModel.from_pretrained(pretrained_dir, local_files_only=True)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Sequential(
            nn.Linear(2304, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

    def forward(self, text_list):
        tokens = self.tokenizer(text_list, return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(self.bert.device) for k, v in tokens.items()}
        out = self.bert(**tokens)
        cls_emb = out.last_hidden_state[:, 0]
        mean_emb = out.last_hidden_state.mean(dim=1)
        max_emb, _ = out.last_hidden_state.max(dim=1)
        concat = torch.cat([cls_emb, mean_emb, max_emb], dim=-1)
        concat = self.dropout(concat)
        return self.proj(concat)

