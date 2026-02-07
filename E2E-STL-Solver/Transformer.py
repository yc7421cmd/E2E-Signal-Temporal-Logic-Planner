# -*- coding: utf-8 -*-
"""Full HKV‑MoE stack: Router, FFN, Decoder layer, Multi‑layer decoder and Trajectory decoder.

Single file so you can *drop‑in replace* the four original classes:
    MoEFFN, MoETransformerDecoderLayer, MoETransformerDecoder, TrajectoryDecoder.

Import path example
-------------------
>>> from model_moe_hkv import HierKVRouter, TrajectoryDecoder_HKV
"""
from __future__ import annotations
import math, random
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def count_expert_usage(topk_indices, num_experts):
    """
    统计一个 batch 中每个 expert 被激活的次数
    输入:
        topk_indices: (B, T, top_k) tensor
        num_experts: int
    输出:
        usage: (num_experts,) tensor，每个 expert 被使用了多少次
    """
    B, T, k = topk_indices.shape
    usage = torch.zeros(num_experts, dtype=torch.int32, device=topk_indices.device)

    for i in range(k):
        idx = topk_indices[:, :, i]  # (B, T)
        flat_idx = idx.reshape(-1)  # 展平为 (B*T,)
        counts = torch.bincount(flat_idx, minlength=num_experts)  # shape: (num_experts,)
        usage += counts

    return usage


# ---------------------------------------------------------------------------
# 0.  Router – Hard bucket + Key‑Value
# ---------------------------------------------------------------------------
class HierKVRouter(nn.Module):
    def __init__(self, n_bucket: int, expert_per_bucket: int, embed_dim: int, k: int = 1, tau: float = 1.0):
        super().__init__()
        self.n_bucket = n_bucket
        self.epb = expert_per_bucket
        self.k = k
        self.tau = tau
        self.expert_key = nn.Parameter(
            torch.randn(n_bucket, expert_per_bucket, embed_dim) / math.sqrt(embed_dim)
        )

    def forward(self, h: torch.Tensor, op_id: torch.Tensor, return_probs: bool = False):
        """
        h: (B,T,C), op_id: (B,T) in [0, n_bucket-1]
        return:
          gid: (B,T,k)  global expert ids
          w  : (B,T,k)  combine weights within top-k (softmax)
          if return_probs: gate_probs (B,T,E_total) one-bucket-filled full probs for regularization
        """
        B, T, C = h.shape
        bucket = op_id.clamp(0, self.n_bucket - 1)      # (B,T)

        # 归一化，避免分数过尖
        h_n = F.normalize(h, dim=-1)
        keys = F.normalize(self.expert_key, dim=-1)     # (Bkt, E', C)

        # 取对应 bucket 的键
        keys_bt = keys[bucket]                           # (B,T,E',C)
        score = (h_n.unsqueeze(-2) * keys_bt).sum(-1)   # (B,T,E')
        score = score / max(self.tau, 1e-6)

        # 全量 bucket 内 softmax 概率（可微）
        probs_local = F.softmax(score, dim=-1)          # (B,T,E')

        # 选 top-k，并把局部 idx 转全局 idx
        topv, local = probs_local.topk(self.k, dim=-1)  # (B,T,k)
        w = topv / (topv.sum(dim=-1, keepdim=True) + 1e-9)
        gid = bucket.unsqueeze(-1) * self.epb + local   # (B,T,k)

        if return_probs:
            E_total = self.n_bucket * self.epb
            gate_probs = h.new_zeros(B, T, E_total)
            base = (bucket * self.epb)                  # (B,T)
            # scatter 概率到全局维度（只填所属 bucket 段）
            idx = base.unsqueeze(-1) + torch.arange(self.epb, device=h.device)  # (B,T,E')
            gate_probs.scatter_(2, idx, probs_local)
            return gid, w, gate_probs
        else:
            return gid, w

# ---------------------------------------------------------------------------
# 1.  FFN with HKV MoE
# ---------------------------------------------------------------------------
class MoEFFN_HKV(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, router: HierKVRouter, drop_p: float = 0.0):
        super().__init__()
        self.router = router
        self.drop_p = drop_p
        n_expert = router.n_bucket * router.epb

        # 共享密集 FFN（保持基线表达）
        self.shared_ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embed_dim)
        )
        # MoE 专家
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embed_dim)
            ) for _ in range(n_expert)
        ])
        # 学习型融合系数：开始时几乎 0，先用 dense FFN 打底，逐步“开闸”MoE
        self.moe_gate_logit = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2)≈0.12，可按需更小

    def forward(self, x: torch.Tensor, op_id: torch.Tensor, aux: Dict[str, torch.Tensor]):
        B, T, C = x.shape
        gid, w, gate_probs = self.router(x, op_id, return_probs=True)  # 可微 gate_probs
        k = gid.size(-1)

        # 展平以便按专家计算
        index_bt = torch.arange(B * T, device=x.device).repeat_interleave(k)
        flat_in  = x.reshape(-1, C)[index_bt]      # (B·T·k , C)
        flat_gid = gid.reshape(-1)                 # (B·T·k)
        flat_w   = w.reshape(-1)                   # (B·T·k)

        out_flat = torch.zeros_like(flat_in)
        uniques = torch.unique(flat_gid[flat_gid >= 0])
        for e in uniques:
            eid = int(e.item())
            sel = (flat_gid == eid)
            out_flat[sel] = self.experts[eid](flat_in[sel]) * flat_w[sel, None]
        moe_out = out_flat.reshape(B, T, k, C).sum(dim=2)

        dense = self.shared_ffn(x)
        alpha = torch.sigmoid(self.moe_gate_logit)       # 0..1
        y = dense + alpha * moe_out                      # 融合输出

        # -------- 正则：负载均衡 + 熵（放进 aux，外层把它加进总 loss）--------
        # importance（Switch/GShard）
        E = gate_probs.size(-1)
        importance = gate_probs.sum(dim=(0, 1))          # (E,)
        imp_frac  = importance / importance.sum().clamp_min(1e-6)
        balance = E * (imp_frac ** 2).sum() - 1.0        # 均匀时≈0
        aux["balance"] = aux.get("balance", x.new_tensor(0.0)) + balance

        # 熵，鼓励不要太尖（可选）
        entropy = -(gate_probs.clamp_min(1e-9) * gate_probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
        aux["router_entropy"] = aux.get("router_entropy", x.new_tensor(0.0)) + (-entropy)  # 惩罚 = -熵

        # 负载日志（可视化）
        with torch.no_grad():
            load = gate_probs.sum(dim=(0, 1))
            load = load / load.sum().clamp_min(1e-6)
            aux["balance_log"] = float(E * (load - 1.0 / E).pow(2).sum())
        aux["last_gid"] = gid.detach()  # (B, T, k)


        return y


# ---------------------------------------------------------------------------
# 2.  Decoder layer
# ---------------------------------------------------------------------------
class MoETransformerDecoderLayer_HKV(nn.Module):
    def __init__(self, embed_dim: int, nhead: int, hidden_dim: int, router: HierKVRouter, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim); self.norm2 = nn.LayerNorm(embed_dim); self.norm3 = nn.LayerNorm(embed_dim)
        self.drop1 = nn.Dropout(dropout); self.drop2 = nn.Dropout(dropout); self.drop3 = nn.Dropout(dropout)
        self.ffn = MoEFFN_HKV(embed_dim, hidden_dim, router)

    def forward(self, tgt, mem, op_id, aux, **kwargs):
        
        
        tgt = tgt + self.drop1(self.self_attn(self.norm1(tgt), self.norm1(tgt), self.norm1(tgt), **kwargs)[0])
        tgt = tgt + self.drop2(self.cross_attn(self.norm2(tgt), mem, mem, **kwargs)[0])
        tgt = tgt + self.drop3(self.ffn(self.norm3(tgt), op_id, aux))
        return tgt

# ---------------------------------------------------------------------------
# 3.  Multi‑layer decoder
# ---------------------------------------------------------------------------
class MoETransformerDecoder_HKV(nn.Module):
    def __init__(self, num_layers: int, embed_dim: int, nhead: int, hidden_dim: int, router: HierKVRouter, moe_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.moe_idx = list(range(num_layers - moe_layers, num_layers))
        self.layers: List[nn.Module] = nn.ModuleList()
        for i in range(num_layers):
            if i in self.moe_idx:
                self.layers.append(MoETransformerDecoderLayer_HKV(embed_dim, nhead, hidden_dim, router, dropout))
            else:
                self.layers.append(nn.TransformerDecoderLayer(embed_dim, nhead, hidden_dim, dropout=dropout, batch_first=True))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tgt, mem, op_id, aux, **kwargs):
        out = tgt
        for i, layer in enumerate(self.layers):
            if i in self.moe_idx:
                out = layer(out, mem, op_id, aux, **kwargs)
            else:
                out = layer(out, mem, **kwargs)
        return self.norm(out)

# ---------------------------------------------------------------------------
# 4.  Trajectory decoder
# ---------------------------------------------------------------------------
class TrajectoryDecoder_HKV(nn.Module):
    def __init__(self, embed_dim: int = 256, hidden_dim: int = 512, seq_len: int = 80, num_layers: int = 4, nhead: int = 8, n_bucket: int = 6, expert_per_bucket: int = 2, moe_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.time_embedding = nn.Parameter(torch.randn(seq_len + 1, embed_dim))
        self.register_buffer("pos_embedding", self._sin_pe(seq_len + 1, embed_dim), False)
        self.frame_embed = nn.Linear(3, embed_dim)
        # router = HierKVRouter(n_bucket, expert_per_bucket, embed_dim, k=1)  # k=1 for single expert per token
        router = HierKVRouter(n_bucket, expert_per_bucket, embed_dim, k=2)


        # Transformer decoder with MoE layers
        self.decoder = MoETransformerDecoder_HKV(num_layers, embed_dim, nhead, hidden_dim, router, moe_layers, dropout)
        self.head = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))

    @staticmethod
    def _sin_pe(L, C):
        pos = torch.arange(L).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, C, 2).float() * (-math.log(10000.0) / C))
        pe = torch.zeros(L, C)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    # -----------------------------------------------------------
    def forward(self, bev_tok, txt_vec,start_pos, pos_emb, struct_tok, op_id, teach_force: float = 0.0, gt: torch.Tensor | None = None):
        # 训练中逐步降低
        device = bev_tok.device
        B = bev_tok.size(0)
        aux: Dict[str, torch.Tensor] = {}
        memory = torch.cat([bev_tok, txt_vec.unsqueeze(1), pos_emb.unsqueeze(1)], 1)
        preds = []
        prev = start_pos.to(device)
        for t in range(1, self.seq_len + 1):
            q = self.frame_embed(prev) + self.time_embedding[t] + self.pos_embedding[t]
            
            if op_id is None:
                slice_op = torch.zeros(B, 1, dtype=torch.long, device=device)
            else:
                t_idx = min(t-1, op_id.size(1)-1)
                slice_op = op_id[:, t_idx:t_idx+1]  # (B,1)

            dec = self.decoder(q.unsqueeze(1),          # tgt  : (B,1,C)
                       memory,                  # mem  : (B,*,C)
                       slice_op,                # op_id: (B,1)
                       aux)
            pred = self.head(dec.squeeze(1))
            preds.append(pred)
            if self.training and gt is not None and random.random() < teach_force:
                nxt = gt[:, t - 1, 1:4]
            else:
                nxt = pred.detach()
            memory = torch.cat([memory, (self.frame_embed(nxt) + self.time_embedding[t] + self.pos_embedding[t]).unsqueeze(1)], 1)
            prev = nxt
        return torch.stack(preds, 1), aux

