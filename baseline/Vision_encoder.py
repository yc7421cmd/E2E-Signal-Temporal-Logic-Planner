import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

# ----------------- 基础：可学习降采样（可选） -----------------
class LearnableDown(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv2d(c, c, 3, stride=2, padding=1, groups=c),  # depthwise
            nn.Conv2d(c, c, 1),                                 # pointwise
            nn.GELU()
        )
    def forward(self, x): 
        return self.op(x)

# ----------------- 核心：潜变量聚合器（Perceiver-style） -----------------
class CrossAttnBlock(nn.Module):
    """ Q(latents) attends to KV(image tokens). batch_first=True """
    def __init__(self, dim, num_heads=8, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, q, kv):
        # q: (B,L,D), kv: (B,N,D)
        qn = self.norm_q(q)
        kvn = self.norm_kv(kv)
        attn_out, _ = self.attn(qn, kvn, kvn, need_weights=False)
        q = q + attn_out
        q = q + self.mlp(q)
        return q

class LatentAggregator(nn.Module):
    """
    将大量 K/V tokens 聚合成少量 latent tokens（L<<N）
    """
    def __init__(self, dim, num_latents=512, num_heads=8, num_layers=2, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * 0.02)
        self.blocks = nn.ModuleList([
            CrossAttnBlock(dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, kv):  # kv: (B,N,D)
        B, N, D = kv.shape
        lat = self.latents.unsqueeze(0).expand(B, -1, -1)  # (B,L,D)
        for blk in self.blocks:
            lat = blk(lat, kv)
        return lat  # (B,L,D)

# ----------------- BEV 编码器 + 潜变量聚合 -----------------
class BEVEncoderMS(nn.Module):
    def __init__(self,
                 out_dim=256,
                 swin_model_name="swin_base_patch4_window7_224",
                 img_size=(320, 480),
                 in_chans=3,
                 pretrained=False,
                 safetensor_path=None,
                 out_indices=(1, 2, 3),
                 unify_level=2,
                 add_coords=True,
                 n_latents=0,                 # 仍可保留全局 learnable tokens（可不启用）
                 # ↓↓↓ 显存与效果的关键开关 ↓↓↓
                 pool_after_fpn: int = 0,     # 2 表示 H/W 再 /2（可选）
                 downsample_mode: str = "avg",# "avg" 或 "learnable"
                 kv_target_tokens: int = 4096,# 对 K/V 自适应到 ~4K（强烈推荐）
                 num_latent_tokens: int = 512,# 输出 L 的大小（建议 256/384/512）
                 num_agg_layers: int = 2,     # cross-attn 层数（1 或 2）
                 agg_heads: int = 8,          # cross-attn 头数
                 agg_mlp_ratio: float = 2.0,
                 agg_dropout: float = 0.0):
        super().__init__()
        self.out_dim = out_dim
        self.unify_level = unify_level
        self.add_coords = add_coords
        self.n_latents = n_latents
        self.out_indices = out_indices

        self.pool_after_fpn = int(pool_after_fpn) if pool_after_fpn else 0
        self.downsample_mode = downsample_mode
        self.kv_target_tokens = int(kv_target_tokens) if kv_target_tokens else 0

        # Backbone
        self.backbone = create_model(
            swin_model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
            in_chans=in_chans,
            img_size=img_size,
            num_classes=0,
        )
        if safetensor_path:
            from safetensors.torch import load_file
            sd = load_file(safetensor_path)
            self.backbone.load_state_dict(sd, strict=False)

        # FPN（延迟构建）
        self._heads_built = False
        self.laterals = None
        self.fpn_convs = None



        # 坐标编码
        extra = 2 if add_coords else 0
        self.coord_proj = nn.Conv2d(out_dim + extra, out_dim, 1)

        # 旧式全局 token（可不启用）
        if n_latents > 0:
            self.global_latents = nn.Parameter(torch.randn(n_latents, out_dim) * 0.02)
        else:
            self.register_parameter('global_latents', None)

        self.norm = nn.LayerNorm(out_dim)

        # learnable 降采样（延迟按C构建）
        self._learnable_down = None

        # 潜变量聚合器（关键）
        self.aggregator = LatentAggregator(
            dim=out_dim,
            num_latents=num_latent_tokens,
            num_heads=agg_heads,
            num_layers=num_agg_layers,
            mlp_ratio=agg_mlp_ratio,
            dropout=agg_dropout,
        )

    def _ensure_nchw(self, feats):
        # 期望通道数（来自 timm 的 feature_info）
        exp_cs = [self.backbone.feature_info[i]["num_chs"] for i in self.out_indices]
        out = []
        for f, c in zip(feats, exp_cs):
            if f.shape[1] == c:              # 已是 NCHW
                out.append(f)
            elif f.shape[-1] == c:           # NHWC -> NCHW
                out.append(f.permute(0, 3, 1, 2).contiguous())
            else:
                raise ValueError(f"Cannot infer channels for {tuple(f.shape)}; expected C={c}")
        return out
    
    # --------- 内部函数 ---------
    def _build_heads(self, feats):
        feats = self._ensure_nchw(feats)     # ★ 关键
        chs = [x.shape[1] for x in feats]
        device = feats[0].device
        self.laterals = nn.ModuleList([nn.Conv2d(c, self.out_dim, 1).to(device) for c in chs])
        self.fpn_convs = nn.ModuleList([nn.Conv2d(self.out_dim, self.out_dim, 3, padding=1).to(device) for _ in chs])
        self._heads_built = True
        if not (0 <= self.unify_level < len(chs)):
            raise ValueError(f"unify_level={self.unify_level} 超出返回尺度数量 {len(chs)}")

    def _fpn(self, feats):
        if not self._heads_built:
            self._build_heads(feats)
        lats = [lat(x) for lat, x in zip(self.laterals, feats)]
        for i in range(len(lats) - 1, 0, -1):
            up = F.interpolate(lats[i], size=lats[i-1].shape[-2:], mode='nearest')
            lats[i-1] = lats[i-1] + up
        outs = [conv(x) for conv, x in zip(self.fpn_convs, lats)]
        return outs

    def _add_coords(self, y):
        if not self.add_coords:
            return y
        B, C, H, W = y.shape
        device = y.device
        gy = torch.linspace(-1, 1, H, device=device)
        gx = torch.linspace(-1, 1, W, device=device)
        yy, xx = torch.meshgrid(gy, gx, indexing='ij')
        coords = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)  # (B,2,H,W)
        y = torch.cat([y, coords], dim=1)
        return self.coord_proj(y)

    def _down_hw(self, y: torch.Tensor) -> torch.Tensor:
        """ 可选：先把 K/V 的 HxW 再小一点（不强制） """
        B, C, H, W = y.shape
        dev = y.device
        if self.pool_after_fpn and self.pool_after_fpn > 1:
            if self.downsample_mode == "learnable":
                if (self._learnable_down is None) or (self._learnable_down.op[0].in_channels != C):
                    self._learnable_down = LearnableDown(C).to(dev)
                s = int(self.pool_after_fpn)
                while s > 1:
                    y = self._learnable_down(y)
                    s //= 2
            else:
                s = int(self.pool_after_fpn)
                y = F.avg_pool2d(y, kernel_size=s, stride=s)
        return y

    def _kv_adapt_tokens(self, y: torch.Tensor) -> torch.Tensor:
        """ 把 K/V token 适配到 ~kv_target_tokens 数量 """
        if self.kv_target_tokens and self.kv_target_tokens > 0:
            B, C, H, W = y.shape
            cur = H * W
            if cur > self.kv_target_tokens:
                r = (cur / self.kv_target_tokens) ** 0.5
                h_new = max(1, int(round(H / r)))
                w_new = max(1, int(round(W / r)))
                y = F.adaptive_avg_pool2d(y, output_size=(h_new, w_new))
        return y

    # --------- 前向 ---------
    def forward(self, bev):  # bev: (B,in_chans,H,W)
        feats = self.backbone(bev)        # list
        feats = self._ensure_nchw(feats)     # ★ 关键

        y     = self._fpn(feats)[self.unify_level]  # (B,C,H,W)
        y     = self._add_coords(y)
        y     = self._down_hw(y)                 # 可选
        y     = self._kv_adapt_tokens(y)         # 强烈推荐：把 K/V 控到 ~4k/8k

        B, C, H, W = y.shape
        kv = y.flatten(2).transpose(1, 2)        # (B,N,C)
        kv = self.norm(kv)

        # 核心：潜变量聚合，输出 L 个 token
        lat_tokens = self.aggregator(kv)         # (B,L,C)

        # 可选：再拼上全局 learnable tokens（通常不需要）
        if self.n_latents and getattr(self, 'global_latents', None) is not None:
            gl = self.global_latents.unsqueeze(0).expand(B, -1, -1)
            lat_tokens = torch.cat([gl, lat_tokens], dim=1)

        return lat_tokens  # (B, L(+g), C)

    @torch.no_grad()
    def warmup(self, C: int, H: int, W: int, device=None):
        if device is None:
            device = next(self.parameters()).device
        dummy = torch.zeros(1, C, H, W, device=device)
        _ = self.forward(dummy)
        torch.cuda.empty_cache()
