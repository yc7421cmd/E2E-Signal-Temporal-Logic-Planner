import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import timm
from transformers import BertTokenizer, BertModel
import os
import json
import numpy as np
from safetensors.torch import load_file
import glob
import re
import time
import matplotlib.pyplot as plt
from trainer import *
import random
from torch.utils.data import Subset
import glob
from robust import *
from STL_encoder import *
from cal_fea_area import *
from stl_filter import *
from traj_filter import *


area = [[-9.5, 9.5],[-9.5, 9.5]]

collision_area1 = [
    [[-6.926746, 3.713431], [2.05, 5.463829]],
    [[-6.926746, 3.713431], [-2.432901, -0.767162]],
    [[-6.926746, 3.713431], [-9.5, -7.030980]],
    [[3.531295, 3.904223], [-9.5, -5.618024]],
    [[5.224491, 9.5], [-5.685480, -5.23987]],
    [[7.65, 9.5], [-9.5, -6.3]],
]


collision_area2 = [
    [[-8.2, 5.16], [2.1, 6.2]],
    [[-8.2, 5.16], [-3.9, -0.25]],
    [[-8.2, 5.16], [-9.5, -6.9]],
    [[3.531295, 3.904223], [-9.5, -5.618024]],
    [[5.224491, 9.5], [-5.685480, -5.23987]],
    [[7.65, 9.5], [-9.5, -6.3]],
]

collision_area3 = [
    [[-9.5, -6.97], [4.16, 9.5]],
    [[-9.5, -6.97], [-9.5, -4.16]],
    [[6.97, 9.5], [-1.1, 9.5]],
    [[3.531295, 3.904223], [-9.5, -5.618024]],
    [[5.224491, 9.5], [-5.685480, -5.23987]],
    [[7.65, 9.5], [-9.5, -6.3]],
]


def find_latest_checkpoint(model_dir="models"):
    # 找所有匹配文件
    ckpt_list = glob.glob(os.path.join(model_dir, "model_epoch_*.pt"))
    if not ckpt_list:
        raise FileNotFoundError("没有找到任何模型文件！")

    # 提取epoch数字
    def extract_epoch(path):
        match = re.search(r"model_epoch_(\d+)\.pt", path)
        return int(match.group(1)) if match else -1

    # 按epoch排序
    ckpt_list.sort(key=extract_epoch)

    latest_ckpt = ckpt_list[-1]
    return latest_ckpt


def find_best_checkpoint(model_dir="models"):
    pt = os.path.join(model_dir, "best_model.pt")
    return pt


def is_within_area(traj):
    x, y = traj[:, 0], traj[:, 1]
    return np.all((area[0][0] <= x) & (x <= area[0][1]) & (area[1][0] <= y) & (y <= area[1][1]))

def is_in_collision(traj, label):
    if(int(label) == 1):
        collision_area = collision_area1
    elif(int(label) == 2):
        collision_area = collision_area2
    else:
        collision_area = collision_area3
    for region in collision_area:
        x_range, y_range = region
        x, y = traj[:, 0], traj[:, 1]
        inside = (x >= x_range[0]) & (x <= x_range[1]) & (y >= y_range[0]) & (y <= y_range[1])
        if np.any(inside):
            return True
    return False

def angle_diff(a, b):
    """将角度差标准化到 [-pi, pi]"""
    diff = a - b
    return (diff + math.pi) % (2 * math.pi) - math.pi


def is_feasible(traj, max_dist=1.65, max_angle_diff=0.75):
    """
    判断整个轨迹是否可行：
    - 相邻点距离不超过 max_dist；
    - 相邻点航向角变化不超过 max_angle_diff。

    参数
    ----
    traj : np.ndarray, shape (T, 3)
        每行是 (x, y, heading)

    返回
    ----
    bool : True 表示轨迹合法
    """
    traj = np.asarray(traj)
    if traj.shape[0] < 2:
        return True  # 只有1个点认为合法

    # 计算相邻点间欧氏距离
    delta_pos = traj[1:, :2] - traj[:-1, :2]  # (T-1, 2)
    dists = np.linalg.norm(delta_pos, axis=1)  # (T-1,)

    # 计算相邻航向角差（考虑角度 wrap-around）
    heading_diff = angle_diff(traj[1:, 2], traj[:-1, 2])

    # heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))  # wrap to [-pi, pi]

    # 判断是否全部满足
    if np.all(dists <= max_dist) and np.all(np.abs(heading_diff) <= max_angle_diff):
        return True
    else:
        return False



def STL_satisfaction(test_path = None):
    import glob
    import json
    import numpy as np

    results = []
    stl_lis = []
    num = 0

    for path in sorted(glob.glob(f"{test_path}/json/sample_*.json")):
        with open(path, 'r') as f:
            data = json.load(f)

        expr = data["stl_formula"]

        # 没有filter
        pred = np.array(data["pred_xy"])
        for_traj = pred[:, 1:4]
        traj = pred[:, 1:3]
        label = data.get("label", 1)  # 默认label为1


        # 有filter
        # pred = np.array(data["pred_xy"])

        # # 可行性微调
        # traj_fil = revise_traj_fil(pred[...,1:4])

        # # STL微调
        # traj_stl = revise_stl(expr, traj_fil)  # [80, 3]

        # label = data.get("label", 1)  # 默认label为1
        # # 碰撞微调
        # traj = revise_traj(expr, traj_stl, label)

        # for_traj = traj
        # traj = traj[...,:2]
        # print("now num is: ", num)


        num += 1
        try:
            # 单条轨迹，单条 STL 公式，放进 batch
            robust = robust_cal([expr], torch.from_numpy(traj).unsqueeze(0))
            # print(robust)
            stl_satisfied = robust > 0
            stl_lis.append(stl_satisfied)
            # print(stl_satisfied)
        except Exception as e:
            stl_satisfied = False
            print(f"[跳过] {path} STL解析失败: {e}")

        # 区域合法性判断
        in_area = is_within_area(traj)
        no_collision = not is_in_collision(traj, label)
        is_feas = is_feasible(for_traj) # 可行

            
        ok = stl_satisfied and in_area and no_collision and is_feas

        results.append({
            "file": path,
            "expr": expr,
            "satisfied": ok,
            "stl": stl_satisfied,
            "in_area": in_area,
            "no_collision": no_collision,
            "feasible": is_feas,
        })

    print(stl_lis[:10])
    

    # 汇总统计
    n_total = len(results)
    n_passed = 0
    n_stl = 0
    n_area = 0
    n_collision = 0
    n_feas = 0
    passed_indices = []

    for idx, r in enumerate(results):
        if r["satisfied"]:
            n_passed += 1
            passed_indices.append(idx)
        if r["stl"]:
            n_stl += 1
        if r["in_area"]:
            n_area += 1
        if r["no_collision"]:
            n_collision += 1
        if r["feasible"]:
            n_feas += 1

    print(f"\n共 {n_total} 条预测，最终通过的有 {n_passed} 条，占比 {n_passed / n_total:.2%}")
    print("-----------------------------------------------------------------------------")
    print(f"其中，满足 STL 的有 {n_stl} 条，占比 {n_stl / n_total:.2%}")
    print(f"区域合法的有 {n_area} 条，占比 {n_area / n_total:.2%}")
    print(f"无碰撞的有 {n_collision} 条，占比 {n_collision / n_total:.2%}")
    print(f"动力学可行的有 {n_feas} 条，占比 {n_feas / n_total:.2%}")
    print("满足 STL 的 sample 索引如下：", passed_indices)



def main():
    parser = argparse.ArgumentParser(description='Vision-based STL Solver')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    args = parser.parse_args()
    if os.path.exists(args.config):
        logger.info(f"加载配置文件: {args.config}")
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        logger.error("配置文件不存在")

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

    batch_size = 4

    # 创建Loader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    n = 800  # 你想要随机选多少个样本
    random.seed(22)  # 固定种子，确保每次结果相同
    indices = random.sample(range(len(test_set)), n)
    test_subset = Subset(test_set, indices)

    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embed_dim = config['embed_dim']
    seq_len = config['seq_len']

    model = MultiModalTrajectoryModel(embed_dim=embed_dim, seq_len=seq_len).to(device)

    save_dir = config['save_dir']
    ten_dir = config['ten_dir']
    test_path = config['test_result']

    trainer = Trainer(model, train_loader, val_loader, test_loader, device, ten_dir, lr=config["lr"])

    # 加载权重
    ckpt_path = find_latest_checkpoint(save_dir)
    # ckpt_path = 'output/models/model_epoch_33.pt'  # now best  new 34  61.50%   
    ckpt = torch.load(ckpt_path, map_location=device)
    model.bev_encoder.warmup(C = 3, H = 480, W = 720, device=device)
    model.load_state_dict(ckpt["model_state_dict"])

    print("加载配置文件：", ckpt_path)
    # test_loss = trainer.test(test_path = test_path)
    # print("testing loss is: ", test_loss)


    t0 = time.time()
    # 计算满足度
    STL_satisfaction(test_path = test_path)
    print("cost time is: ", time.time() - t0)


if __name__ == "__main__":
    main()
