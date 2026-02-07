import torch
import torch.nn as nn
import random


class TrajectoryDecoder(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=512, seq_len=80, num_layers=4, nhead=8, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        self.time_embedding = nn.Parameter(torch.randn(seq_len + 1, embed_dim))
        self.pos_embedding = nn.Parameter(self._get_sinusoid_encoding(seq_len + 1, embed_dim), requires_grad=False)

        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )

        self.frame_embed = nn.Linear(3, embed_dim)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

    def _get_sinusoid_encoding(self, seq_len, dim):
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe = torch.zeros(seq_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, bev_tokens, text_emb, pose_emb, teaching_force, gt=None, start_pos=None):
        B, C = text_emb.shape
        device = bev_tokens.device

        # 初始 memory（B, N+2, C）
        memory = torch.cat([bev_tokens, text_emb.unsqueeze(1), pose_emb.unsqueeze(1)], dim=1)  # (B, S0, C)

        preds = []
        prev_pred = start_pos.to(device)

        for t in range(1, self.seq_len + 1):
            # 构造 decoder input（当前要预测一个 token）
            query_input = self.frame_embed(prev_pred) + self.time_embedding[t] + self.pos_embedding[t]
            query_input = query_input.unsqueeze(1)  # (B, 1, C)

            # Transformer 解码器调用
            decoded = self.transformer(
                src=memory,       # 当前 memory 包含先验信息 + 历史预测
                tgt=query_input   # 当前时刻 query
            )  # (B, 1, C)

            decoded = decoded.squeeze(1)
            pred = self.head(decoded)  # (B, 3)
            preds.append(pred)

            # 下一个 prev_pred
            if gt is not None and torch.rand(1).item() < teaching_force:
                next_input = gt[:, t - 1, 1:4].detach() + torch.randn_like(gt[:, t - 1, 1:4]) * 0.02
            else:
                next_input = pred.detach()

            # 把当前预测加入 memory
            new_mem = self.frame_embed(next_input) + self.time_embedding[t] + self.pos_embedding[t]
            new_mem = new_mem.unsqueeze(1)  # (B,1,C)
            memory = torch.cat([memory, new_mem], dim=1)

            prev_pred = next_input

        pred_seq = torch.stack(preds, dim=1)  # (B, T, 3)
        return pred_seq


