import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import ViTFeatureExtractor, ViTModel
from timm import create_model
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
import os
import math
import random
from tqdm import tqdm
import warnings
from torch.utils.tensorboard import SummaryWriter
import time
import requests
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch.nn.functional as F
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import glob
import torch
from torch.utils.data import DataLoader
import timm
import torch
# 这里手动加载本地权重
from safetensors.torch import load_file
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import yaml
import logging
import math
import random
from STL_encoder import *
from Pose_region_encoder import *
from Vision_encoder import *
from Transformer import *
from torch.utils.tensorboard import SummaryWriter
from robust import *

# 预先定义的场景中可能出现的环境(得改一下)
"""
scene1:
1:A
2:B
3:C
4:D
"""

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

obstacles = [
    [[-8, 4.36], [2.1, 6.2]],
    [[-8, 4.36], [-3.9, -0.25]],
    [[-8, 4.36], [-9.5, -6.9]],
    [[3.28, 4.22], [-9.5, -5.5]],
    [[4.97, 9.5], [-6.0, -4.9]],
    [[7.65, 9.5], [-9.5, -6.3]]
]



import torch.nn.functional as F

def trajectory_obstacle_penalty(traj: torch.Tensor, obstacles: list, penalty_weight=10.0):
    """
    traj: (B, T, 2)
    obstacles: list of [[xmin, xmax], [ymin, ymax]]
    """
    B, T, _ = traj.shape
    device = traj.device

    penalty_mask = torch.zeros((B, T), dtype=torch.bool, device=device)

    for obs in obstacles:
        (x_min, x_max), (y_min, y_max) = obs
        x_in = (traj[..., 0] >= x_min) & (traj[..., 0] <= x_max)
        y_in = (traj[..., 1] >= y_min) & (traj[..., 1] <= y_max)
        inside = x_in & y_in  # (B, T)
        penalty_mask |= inside  # 保持为 bool 类型

    # 每个 batch 进入障碍的比例
    penalty = penalty_mask.float().mean(dim=1)  # 转为 float 再求均值
    return penalty_weight * penalty.mean()


def angle_diff_torch(a, b):
    # wrap 到 [-pi, pi]，可微
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))

def feasibility_penalty(traj, max_dist=1.5, max_angle_diff=0.6, reduction="mean",
                        use_l2=True, margin_dist=0.0, margin_angle=0.0):
    """
    traj: (B, T, 3) -> (x, y, heading[rad])
    返回：标量损失
    """
    B, T, D = traj.shape
    assert D >= 3, "traj 的最后维需要至少包含 (x, y, heading)"

    # 相邻位移 & 距离
    delta_pos = traj[:, 1:, :2] - traj[:, :-1, :2]          # (B, T-1, 2)
    dists = torch.linalg.norm(delta_pos, dim=-1)            # (B, T-1)

    # 相邻航向差（考虑环绕）
    dtheta = angle_diff_torch(traj[:, 1:, 2], traj[:, :-1, 2]).abs()  # (B, T-1)

    # 仅对“超出阈值”的部分计罚（hinge）
    dist_violation  = F.relu(dists  - (max_dist  - margin_dist))
    angle_violation = F.relu(dtheta - (max_angle_diff - margin_angle))

    if use_l2:
        dist_pen  = dist_violation.pow(2)
        angle_pen = angle_violation.pow(2)
    else:
        dist_pen  = dist_violation
        angle_pen = angle_violation

    feas_pen = dist_pen + angle_pen  # (B, T-1)

    if reduction == "mean":
        return feas_pen.mean()
    elif reduction == "sum":
        return feas_pen.sum()
    else:  # 'none'
        return feas_pen


# 加载 处理数据
class GazeboDataset(Dataset):

    def __init__(self, data_root):
        self.data_root = data_root
        self.samples = sorted(
            [os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]

        # Load BEV
        bev_path = os.path.join(sample_dir, "bev.npy")
        bev = np.load(bev_path)
        bev_tensor = torch.tensor(bev, dtype=torch.float32).permute(2, 0, 1)  # HWC->CHW

        # Load trajectory
        traj_path = os.path.join(sample_dir, "trajectory.csv")
        traj_df = pd.read_csv(traj_path)

        # Process trajectory
        traj_np = traj_df.values  # (N,6)
        n_steps = traj_np.shape[0]
        target_steps = 80

        if n_steps < target_steps:
            # Get the last row
            last_row = traj_np[-1].copy()
            # Set v, w = 0
            last_row[4] = 0.0
            last_row[5] = 0.0
            # Repeat as needed
            n_pad = target_steps - n_steps
            pad_rows = np.tile(last_row, (n_pad, 1))
            traj_np = np.vstack([traj_np, pad_rows])

        elif n_steps > target_steps:
            # 如果比目标长，就截断
            traj_np = traj_np[:target_steps]

        traj_tensor = torch.tensor(traj_np, dtype=torch.float32)

        # Load meta
        meta_path = os.path.join(sample_dir, "meta.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        return {"bev": bev_tensor, "trajectory": traj_tensor, "meta": meta}


def collate_fn(batch):
    # Batch is a list of dicts
    bev_batch = torch.stack([item["bev"] for item in batch])  # B,C,H,W
    traj_batch = torch.stack([item["trajectory"] for item in batch])  # B,N,6
    meta_batch = [item["meta"] for item in batch]

    return {
        "bev": bev_batch,
        "trajectory": traj_batch,  # 因为轨迹长度可能不一样，这里保留list
        "meta": meta_batch
    }


# 集成了四个类 只负责前向传播
class MultiModalTrajectoryModel(nn.Module):

    def __init__(self, embed_dim=256, seq_len=80):
        super().__init__()

        # BEV Encoder
        self.bev_encoder = BEVEncoderMS(
            out_dim=embed_dim,            # 和你下游 token 维度一致
            swin_model_name="swin_base_patch4_window7_224",
            in_chans= 3,        # 例如 60
            img_size=(480, 720),
            pretrained=False,
            safetensor_path=
            "/root/project/final_project-2/baseline/hugging_face/models--timm--swin_base_patch4_window7_224.ms_in22k_ft_in1k/snapshots/a6a1eb2321b4f556fa0fa243fb777d47679f13c9/model.safetensors",
            out_indices=(1, 2, 3),        # 按需
            unify_level=2,
            add_coords=True,
            # ↓ 关键
            pool_after_fpn=0,            # 先不额外 /2
            kv_target_tokens=4096,       # 把 K/V 控到 ~4K
            num_latent_tokens=512,       # 输出 512 个 token
            num_agg_layers=2,            # 两层 cross-attn
            agg_heads=8,
        )


        # Text Encoder
        self.text_encoder = TextEncoder(
            pretrained_dir=
            "/root/project/final_project-2/baseline/hugging_face/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594",
            embed_dim=embed_dim)

        # Pose + Region Encoder
        self.pose_encoder = PoseRegionEncoder(embed_dim=embed_dim)

        # Trajectory Decoder

        self.decoder = TrajectoryDecoder(embed_dim=embed_dim, hidden_dim=128, seq_len=80, num_layers=4, nhead=4)

    def forward(self, bev, text_list, pose, region, epoch, gt):
        """
        bev: (B,3,H,W)
        text_list: List[str]
        pose: (B,3)
        region: (B,4)
        """
        bev_tokens = self.bev_encoder(bev)  # (B,N,embed_dim)
        text_emb = self.text_encoder(text_list)  # (B,embed_dim)
        pose_emb, pose = self.pose_encoder(pose, region)  # (B,embed_dim)

        pred_traj = self.decoder(bev_tokens, text_emb, pose_emb, epoch, gt, pose)  # (B,80,3)

        return pred_traj


class Trainer:

    def __init__(self, model, train_loader, val_loader, test_loader, device, ten_dir, lr=1e-4, total_epoch=100):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=1e-2  # 比Adam更容易用稍大的weight decay
        )

        self.writer = SummaryWriter(log_dir=ten_dir)
        self.epoch = 0
        self.total_epoch = total_epoch
        self.global_step = 0  # 在 __init__ 或训练开始时加上这行
        self.val_step = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_steps = 8000   # 假设总的epoch大约8000步
        progress_bar = tqdm(self.train_loader, total=len(self.train_loader), desc="Training")

        for batch in progress_bar:

            bev = batch["bev"].to(self.device)
            # bev = F.interpolate(bev, size=(480, 736), mode="bilinear", align_corners=False)
            traj_gt = batch["trajectory"].to(self.device)


            texts = []
            stl = []  # [B]
            start_pos = []
            for m in batch["meta"]:
                expr = m["stl_formula"]
                stl.append(expr)
                start_pos.append(m['start_pose'])
                # 使用递归解析并结构化表达式
                try:
                    ast = parse_full_stl(expr)
                    text = ast_to_semantic_text(ast)
                except Exception as e:
                    print(f"[STL解析失败] {expr} 错误信息: {e}")
                    text = "op always start 0 end 0 predicate fail relation = value fail"  # fallback
                texts.append(text)

            pose = torch.tensor(
                [[m["start_pose"]["x"], m["start_pose"]["y"], m["start_pose"]["yaw"]] for m in batch["meta"]],
                dtype=torch.float32).to(self.device)

            region = torch.tensor([[
                m["region_bounds"]["x_min"], m["region_bounds"]["y_min"], m["region_bounds"]["x_max"],
                m["region_bounds"]["y_max"]
            ] for m in batch["meta"]],
                                  dtype=torch.float32).to(self.device)

            self.optimizer.zero_grad()


            teaching_force = float((self.total_epoch - self.epoch) / self.total_epoch)
            if(self.total_epoch < 15):  # 控制再训练的teaching_force在0.2以下
                teaching_force *= 0.8
            teaching_force = max(0, teaching_force)

            pred_traj = self.model(bev, texts, pose, region, teaching_force, traj_gt)

            gt = traj_gt[..., 1:4].to(self.device)
            mse_loss = self.loss_fn(pred_traj, gt)
            end_loss = F.mse_loss(pred_traj[:, -1, :2], gt[:, -1, 1:3])

            # end_loss = F.mse_loss(pred_traj[:, -1, :2], gt[:, -1, :2])
            # print(pred_traj)
            obstacle = trajectory_obstacle_penalty(pred_traj[:, :, :2], obstacles)  

            # 计算robust,0也加进去
            B = pred_traj.shape[0]
            start_xy = torch.tensor([[p["x"], p["y"], p['yaw']] for p in start_pos], dtype=pred_traj.dtype, device=pred_traj.device)  # (B, 3)
            start_xy = start_xy[:, None, :]  # (B, 1, 3)
            pred_traj = torch.cat([start_xy, pred_traj], dim=1)  # (B, T+1, 2)

            robust_loss = robust_cal(stl, pred_traj[:, :, :2])

            robust_penalty = min(robust_loss, 0.0) * (-1)# 仅惩罚 <0 的情况

            # 可行性 loss
            feas_loss  = feasibility_penalty(pred_traj)
            # print("feas_loss is: ", feas_loss)

            

            # 
            r = min(1.0, max(self.global_step / total_steps, 0.3))  # 线性增大到 1.0 从 0.3 开始

            w_mse  = 1.5
            w_rho  = 1.5 * r                  # ρ 不足才惩罚，权重不必太大
            w_obs  = 8.0 * r                  # 显著提高障碍物权重（你现在 1.5 偏低）
            w_feas = 4.0 * r                  # 同理，提高动力学可行性权重

            # 总loss,之前训练的没有end_loss
            loss = w_mse * mse_loss - w_rho * robust_penalty + w_obs * obstacle + w_feas * feas_loss

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            self.writer.add_scalar("Loss/Train", loss.item(), self.global_step)
            self.global_step += 1
            # 显示loss
            progress_bar.set_postfix(loss=f"{loss:.4f}")


        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0

        for batch in tqdm(self.val_loader, total=len(self.val_loader), desc="Valing"):

            bev = batch["bev"].to(self.device)
            # bev = F.interpolate(bev, size=(480, 736), mode="bilinear", align_corners=False)
            traj_gt = batch["trajectory"].to(self.device)
            texts = []
            stl = []  # [B]
            start_pos = []
            for m in batch["meta"]:
                expr = m["stl_formula"]
                start_pos.append(m['start_pose'])
                stl.append(expr)

                # 使用递归解析并结构化表达式
                try:
                    ast = parse_full_stl(expr)
                    text = ast_to_semantic_text(ast)
                except Exception as e:
                    print(f"[STL解析失败] {expr} 错误信息: {e}")
                    text = "op always start 0 end 0 predicate fail relation = value fail"  # fallback
                texts.append(text)

            pose = torch.tensor(
                [[m["start_pose"]["x"], m["start_pose"]["y"], m["start_pose"]["yaw"]] for m in batch["meta"]],
                dtype=torch.float32).to(self.device)

            region = torch.tensor([[
                m["region_bounds"]["x_min"], m["region_bounds"]["y_min"], m["region_bounds"]["x_max"],
                m["region_bounds"]["y_max"]
            ] for m in batch["meta"]],
                                  dtype=torch.float32).to(self.device)

            pred_traj = self.model(bev, texts, pose, region, 0.0, traj_gt)  # 别太差了
            mse_loss = self.loss_fn(pred_traj, traj_gt[..., 1:4])
            end_loss = F.mse_loss(pred_traj[:, -1, :2], traj_gt[:, -1, 1:3])
            
            # obstacle_penalty = trajectory_obstacle_penalty(pred_traj[:, :, :2], obstacles)

            # # 计算robust,0也加进去
            B = pred_traj.shape[0]
            start_xy = torch.tensor([[p["x"], p["y"], p['yaw']] for p in start_pos], dtype=pred_traj.dtype, device=pred_traj.device)  # (B, 3)
            start_xy = start_xy[:, None, :]  # (B, 1, 3)
            pred_traj = torch.cat([start_xy, pred_traj], dim=1)  # (B, T+1, 2)

            
            robust_loss = robust_cal(stl, pred_traj[:, :, :2])


            # # 总loss
            # loss = 1.2 * mse_loss - 2 * robust_loss + 3 * obstacle_penalty
            # 总loss
            loss = mse_loss + 0.5 * end_loss - robust_loss

            total_loss += loss.item()
            self.writer.add_scalar("Loss/Val", loss.item(), self.val_step)
            self.val_step += 1

        return total_loss / len(self.val_loader)

    @torch.no_grad()
    def test(self, test_loader=None, test_path = None):
        test_loader = test_loader if test_loader is not None else self.test_loader

        self.model.eval()
        total_loss = 0.0

        for i, batch in tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc="Testing"):

            bev = batch["bev"].to(self.device)
            # bev = F.interpolate(bev, size=(480, 736), mode="bilinear", align_corners=False)
            traj_gt = batch["trajectory"].to(self.device)

            texts = []
            stl = []  # [B]
            start_pos = []
            for m in batch["meta"]:
                start_pos.append(m['start_pose'])
                expr = m["stl_formula"]
                stl.append(expr)

                # 使用递归解析并结构化表达式
                try:
                    ast = parse_full_stl(expr)
                    text = ast_to_semantic_text(ast)
                except Exception as e:
                    print(f"[STL解析失败] {expr} 错误信息: {e}")
                    text = "op always start 0 end 0 predicate fail relation = value fail"  # fallback
                texts.append(text)

            pose = torch.tensor(
                [[m["start_pose"]["x"], m["start_pose"]["y"], m["start_pose"]["yaw"]] for m in batch["meta"]],
                dtype=torch.float32).to(self.device)

            region = torch.tensor([[
                m["region_bounds"]["x_min"], m["region_bounds"]["y_min"], m["region_bounds"]["x_max"],
                m["region_bounds"]["y_max"]
            ] for m in batch["meta"]],
                                  dtype=torch.float32).to(self.device)

            pred_traj = self.model(bev, texts, pose, region, 0.0, traj_gt)

            mse_loss = self.loss_fn(pred_traj, traj_gt[..., 1:4])

            end_loss = F.mse_loss(pred_traj[:, -1, :2], traj_gt[:, -1, 1:3])

            B = pred_traj.shape[0]
            start_xy = torch.tensor([[p["x"], p["y"], p['yaw']] for p in start_pos], dtype=pred_traj.dtype, device=pred_traj.device)  # (B, 3)
            start_xy = start_xy[:, None, :]  # (B, 1, 3)
            pred_traj = torch.cat([start_xy, pred_traj], dim=1)  # (B, T+1, 2) B = 1
            store_xy = pred_traj[0, :, :3].cpu().numpy()

            
            robust_loss = robust_cal(stl, pred_traj[:, :, :2])

            # print(pred_traj)

            # 总loss
            loss = mse_loss + 0.5 * end_loss - robust_loss

            total_loss += loss.item()

            # 可视化得留着
            x_min = batch["meta"][0]["region_bounds"]["x_min"]
            x_max = batch["meta"][0]["region_bounds"]["x_max"]
            y_min = batch["meta"][0]["region_bounds"]["y_min"]
            y_max = batch["meta"][0]["region_bounds"]["y_max"]

            # 把预测和gt先取前两个维度
            pred_xy = pred_traj[0, :, :2].cpu().numpy()
            gt_xy = traj_gt[0, :, 1:3].cpu().numpy()

            # BEV图
            bev_np = batch["bev"][0].cpu().numpy().transpose(1, 2, 0)
            bev_norm = (bev_np - bev_np.min()) / (bev_np.max() - bev_np.min() + 1e-8)
            H, W = bev_norm.shape[:2]

            # 坐标映射函数
            def world_to_pixel(xy):
                x, y = xy[:, 0], xy[:, 1]
                u = (x - x_min) / (x_max - x_min) * W
                v = (y_max - y) / (y_max - y_min) * H
                return np.stack([u, v], axis=1)

            # 映射轨迹
            pred_xy_px = world_to_pixel(pred_xy)
            gt_xy_px = world_to_pixel(gt_xy)

            start_x = batch["meta"][0]["start_pose"]["x"]
            start_x = (start_x - x_min) / (x_max - x_min) * W

            start_y = batch["meta"][0]["start_pose"]["y"]
            start_y = (y_max - start_y) / (y_max - y_min) * H

            plt.figure(figsize=(6, 6))
            plt.imshow(bev_norm)

            # 绘制
            plt.plot(pred_xy_px[:, 0], pred_xy_px[:, 1], 'r-', label='Predicted')
            plt.plot(gt_xy_px[:, 0], gt_xy_px[:, 1], 'g--', label='Ground Truth')
            plt.scatter(start_x, start_y, c='blue', s=5, label='Start')

            plt.title(expr, fontsize=8)
            plt.legend()
            plt.axis("off")

            os.makedirs(test_path, exist_ok=True)
            os.makedirs(f"{test_path}/shown", exist_ok=True)
            os.makedirs(f"{test_path}/json", exist_ok=True)
            save_path = os.path.join(f"{test_path}/shown", f"sample_{i:03d}.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
            # 保存轨迹和 STL 信息
            # 假设每步时间为 0.1 秒（你可以按实际情况改）
            dt = 1
            timesteps = [round(i * dt, 2) for i in range(len(pred_xy))]  # e.g., [0.0, 0.1, ..., 1.9]

            # 将 pred_xy 每一项扩展为 [t, x, y]
            pred_xy_with_time = [[float(t), float(x), float(y), float(yaw)] for (t, (x, y, yaw)) in zip(timesteps, store_xy)]

            # 保存
            save_data = {
                "stl_formula": expr,
                "start_pose": batch["meta"][0]["start_pose"],      # dict with x, y, yaw
                "pred_xy": pred_xy_with_time,                      # List of [t, x, y]
                "label": batch["meta"][0].get('label', 1),  # 默认label为1
            }
            # 保存为 JSON
            json_save_path = os.path.join(f"{test_path}/json", f"sample_{i:03d}.json")
            with open(json_save_path, 'w') as f:
                json.dump(save_data, f, indent=2)


        return total_loss / len(self.test_loader)


def main():
    parser = argparse.ArgumentParser(description='Vision-based STL Solver')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch_size', type=int, help='批次大小')
    args = parser.parse_args()
    if os.path.exists(args.config):
        logger.info(f"加载配置文件: {args.config}")
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        logger.error("配置文件不存在")
    if args.epochs:
        config['max_epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size

    logger.info(f"配置: {config}")

    DATA_ROOT = config['data_root']
    dataset = GazeboDataset(DATA_ROOT)

    # 计算长度
    total_len = len(dataset)
    train_len = int(total_len * 0.8)
    val_len = int(total_len * 0.1)
    test_len = total_len - train_len - val_len  # 剩下的都放到test

    print(f"Total samples: {total_len}")
    print(f"Train: {train_len}, Val: {val_len}, Test: {test_len}")

    # 随机划分
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(42)  # 固定随机种子，保证可复现
    )
    batch_size = config['batch_size']
    # 创建Loader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = config['save_dir']
    ten_dir = config['ten_dir']

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(ten_dir, exist_ok=True)

    embed_dim = config['embed_dim']
    seq_len = config['seq_len']

    model = MultiModalTrajectoryModel(embed_dim=embed_dim, seq_len=seq_len).to(device)

    trainer = Trainer(model,
                      train_loader,
                      val_loader,
                      test_loader,
                      device,
                      ten_dir,
                      lr=config["lr"],
                      total_epoch=config["max_epochs"])

    best_val_loss = float("inf")

    # Train Loop
    for epoch in range(config["max_epochs"] + config['last_epoch']):
        train_loss = trainer.train_epoch()
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}")

        save_epoch = config["save_epoch"]
        val_epoch = config["val_epoch"]

        if ((epoch + 1) % save_epoch == 0):
            model_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'epoch': epoch + 1,
                'loss': train_loss
                }, model_path)
            print(f"Saved checkpoint: {model_path}")

            # 🔽 保留最近5个模型，其余删除
            model_files = sorted(
                [f for f in os.listdir(save_dir) if f.startswith("model_epoch_") and f.endswith(".pt")],
                key=lambda x: int(x.split("_")[-1].split(".")[0])
            )
            if len(model_files) > 5:
                to_delete = model_files[:-5]
                for fname in to_delete:
                    fpath = os.path.join(save_dir, fname)
                    os.remove(fpath)
                    print(f"Deleted old checkpoint: {fpath}")

        if ((epoch + 1) % val_epoch == 0):
            val_loss = trainer.validate()
            print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(save_dir, "best_model.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'loss': val_loss
                    }, best_path)
                print(f"New best model saved to: {best_path}")
        trainer.epoch += 1


if __name__ == "__main__":
    main()
