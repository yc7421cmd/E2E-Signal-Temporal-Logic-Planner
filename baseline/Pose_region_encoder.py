import torch
import torch.nn as nn


class PoseRegionEncoder(nn.Module):

    def __init__(self, embed_dim=256, hidden_dim=128):
        super().__init__()

        # Pose Encoder
        self.pose_mlp = nn.Sequential(nn.Linear(3, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, embed_dim))

        # Region Encoder
        self.region_mlp = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU(inplace=True),
                                        nn.Linear(hidden_dim, embed_dim))

        # 融合投射
        self.fuse_proj = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, pose, region):
        """
        pose: (B,3)
        region: (B,4)
        """
        pose_emb = self.pose_mlp(pose)  # (B,embed_dim)
        region_emb = self.region_mlp(region)  # (B,embed_dim)

        fused = torch.cat([pose_emb, region_emb], dim=-1)  # (B,2*embed_dim)
        fused = self.fuse_proj(fused)  # (B,embed_dim)

        return fused, pose

