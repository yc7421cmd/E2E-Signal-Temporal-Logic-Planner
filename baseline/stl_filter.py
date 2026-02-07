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
from torch.utils.tensorboard import SummaryWriter
import math
import numpy as np
from test import *



area = [[-9.5, 9.5],[-9.5, 9.5]]

collision_area = [
    [[-6.926746, 3.713431], [2.05, 5.463829]],
    [[-6.926746, 3.713431], [-2.432901, -0.767162]],
    [[-6.926746, 3.713431], [-9.5, -7.030980]],
    [[3.531295, 3.904223], [-9.5, -5.618024]],
    [[5.224491, 9.5], [-5.685480, -5.23987]],
    [[7.65, 9.5], [-9.5, -6.3]],
]


def is_valid_sample(sample, point_e, area, collision_area, max_dist=1.65, max_delta_heading=0.75):
    """
    判断单个采样点是否满足以下条件：
    1. 在 area 内；
    2. 不落入任何 collision_area；
    3. 与 point_e 距离 ≤ max_dist；
    4. 与 point_e 的 heading 差 ≤ max_delta_heading。
    
    参数
    ----
    sample : (3,) array-like
        [x, y, heading]
    point_e : (3,) array-like
        参考点 [x, y, heading]
    area : [[x_min, x_max], [y_min, y_max]]
    collision_area : list of [[x_min, x_max], [y_min, y_max]]
    max_dist : float
        最大允许距离（默认 2.0）
    max_delta_heading : float
        最大允许角度变化（弧度，默认 0.5）

    返回
    ----
    bool : True 表示合法，False 表示非法
    """
    # print(sample)
    x = sample[0]
    y = sample[1]
    theta = sample[2]

    x_ref = point_e[0]
    y_ref = point_e[1]
    theta_ref = point_e[2]

    # ① 区域合法性
    (x_min, x_max), (y_min, y_max) = area
    if not (x_min <= x <= x_max and y_min <= y <= y_max):
        return False

    # ② 碰撞检测
    for (x_rng, y_rng) in collision_area:
        if x_rng[0] <= x <= x_rng[1] and y_rng[0] <= y <= y_rng[1]:
            return False

    # ③ 距离限制
    dist = np.sqrt((x - x_ref) ** 2 + (y - y_ref) ** 2)
    if dist > max_dist:
        return False

    # ④ 航向角限制（用 wrap-around 差）
    delta_theta = np.arctan2(np.sin(theta - theta_ref), np.cos(theta - theta_ref))
    if abs(delta_theta) > max_delta_heading:
        return False

    return True


def analyze_failed_stl(expr, traj_f):
    """
    输入：
        expr: STL表达式（str）
        traj_f: (T, D) tensor，表示轨迹，至少前两维是 x, y
    输出：
        failed_tasks: list of dict，仅包含不满足（ρ<0）的原子子任务
    """
    traj = traj_f[..., :2]  # 只用 x, y
    ast = parse_full_stl(expr)
    root_rho = eval_ast(ast, traj).item()

    def find_failed_subtask(ast_node, traj):
        if "type" in ast_node and ast_node["type"] in ("and", "or"):
            left_ast = ast_node["left"]
            right_ast = ast_node["right"]
            left_rho = eval_ast(left_ast, traj).item()
            right_rho = eval_ast(right_ast, traj).item()

            if ast_node["type"] == "and":
                result = []
                if left_rho < 0:
                    result += find_failed_subtask(left_ast, traj)
                if right_rho < 0:
                    result += find_failed_subtask(right_ast, traj)
                return result

            elif ast_node["type"] == "or":
                if left_rho >= 0 or right_rho >= 0:
                    # 有一边满足就算满足，不返回任何子任务
                    return []
                else:
                    # 两边都不满足，返回ρ更大的一边
                    if left_rho <= right_rho:
                        return find_failed_subtask(right_ast, traj)
                    else:
                        return find_failed_subtask(left_ast, traj)

        else:
            op = ast_node['op']
            start = int(ast_node["start"])
            end = int(ast_node["end"])
            region = ast_node["value"]
            region_bounds = region_dict1[region]

            rho_list = []
            for i in range(start, end + 1):
                x, y = traj[i, 0], traj[i, 1]
                delta_step = [
                    x - region_bounds['x_min'],
                    region_bounds['x_max'] - x,
                    y - region_bounds['y_min'],
                    region_bounds['y_max'] - y
                ]
                rho_i = smooth_min(delta_step)
                rho_list.append(rho_i)

            if op == "eventually":
                rho = smooth_max(rho_list)
                best_t = int(torch.argmax(torch.tensor(rho_list)))
                result = {
                    "type": "F",
                    "start": start,
                    "end": end,
                    "region": region,
                    "rho": rho.item(),
                    "max_rho_time": start + best_t
                }
            elif op == "always":
                rho = smooth_min(rho_list)
                violating_times = [start + i for i, v in enumerate(rho_list) if v.item() < 0]
                result = {
                    "type": "G",
                    "start": start,
                    "end": end,
                    "region": region,
                    "rho": rho.item(),
                    "violating_times": violating_times
                }
            else:
                raise ValueError(f"未知操作符: {op}")

            if result["rho"] < 0:
                return [result]
            else:
                return []

    return find_failed_subtask(ast, traj)


def in_region(region, point):
    region_bounds = region_dict1[region]
    x_min = region_bounds['x_min']
    x_max = region_bounds['x_max']
    y_min = region_bounds['y_min']
    y_max = region_bounds['y_max']
    if(point[0] >= x_min and point[0] <= x_max and point[1] >= y_min and point[1] <= y_max):
        return True
    return False


def RRT_stl_points(point_s, point_e, region, max_dist=1.65, max_delta_theta=0.75, num_samples=1, rng=None, num_times = 100):

    flag = False
    time = 0
    while(flag == False and time <= num_times):
        if rng is None:
            rng = np.random.default_rng()

        x, y, heading = map(float, point_s)

        # 采样距离 r ∈ (0, max_dist]（可选均匀也可选指数 / 高斯，这里用均匀）
        r = rng.uniform(low=0.0, high=max_dist, size=num_samples)

        # 采样偏转角 Δθ ∈ [-max_delta_theta, +max_delta_theta]
        delta_theta = rng.uniform(low=-max_delta_theta, high=max_delta_theta, size=num_samples)

        # 根据极坐标计算新位置
        new_heading = heading + delta_theta
        new_x = x + r * np.cos(new_heading)
        new_y = y + r * np.sin(new_heading)

        samples = np.stack([new_x, new_y, new_heading], axis=1)
        samples = samples[0]
        time += 1
        # 也检查了可行性
        if(is_valid_sample(samples, point_e, area, collision_area) and in_region(region, samples)):
            flag = True
    if(flag == False):
        return [-1, -1, -1]

    return samples


def revise_stl(expr, traj, max_samples=10):

    xy_traj = traj[..., :2]

    robust = robust_cal([expr], torch.from_numpy(xy_traj).unsqueeze(0))
    # print(robust)
    stl_satisfied = robust > 0
    time = 0
    while(not stl_satisfied and time <= max_samples):
        time += 1
        failed_dict = analyze_failed_stl(expr, traj)
        for task in failed_dict:
            if task["type"] == "F":
                region = task["region"]
                time_step = task['max_rho_time']
                point_s = traj[time_step - 1] if time_step > 0 else traj[time_step]  # 用前一帧作为参考（或当前）
                point_e = traj[time_step + 1] if time_step < traj.shape[0] - 1 else traj[time_step]
                sample = RRT_stl_points(point_s=point_s, point_e=point_e,region = region, num_samples=1)
                if(sample[0] != -1):
                    traj[time_step] = sample

            elif(task["type"] == "G"):
                region = task["region"]
                time_list = task["violating_times"]
                for time_step in time_list:
                    point_s = traj[time_step - 1] if time_step > 0 else traj[time_step]  # 用前一帧作为参考（或当前）
                    point_e = traj[time_step + 1] if time_step < traj.shape[0] - 1 else traj[time_step]
                    sample = RRT_stl_points(point_s=point_s, point_e=point_e,region = region, num_samples=1)
                    if(sample[0] != -1):
                        traj[time_step] = sample

        
        xy_traj = traj[..., :2]
        robust = robust_cal([expr], torch.from_numpy(xy_traj).unsqueeze(0))
        # print(robust)
        stl_satisfied = robust > 0
    
    return traj



# [
#   {
#     "type": "G",
#     "start": 6,
#     "end": 12,
#     "region": "B",
#     "rho": -0.25,
#     "violating_times": [6,7,8]
#   },
#   {
#     "type": "F",
#     "start": 13,
#     "end": 15,
#     "region": "A",
#     "rho": -0.12,
#     "max_rho_time": 14
#   }
# ]

