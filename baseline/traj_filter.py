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
from cal_fea_area import *


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


def angle_diff(a, b):
    """将角度差标准化到 [-pi, pi]"""
    diff = a - b
    return (diff + math.pi) % (2 * math.pi) - math.pi

def is_feasible_point(point_s, curr, max_dist=1.65, max_angle_diff=0.75):
    # 自动转换为 tensor
    point_s = torch.as_tensor(point_s).float()
    curr = torch.as_tensor(curr).float()

    # 取坐标
    dx = curr[0] - point_s[0]
    dy = curr[1] - point_s[1]
    dist = torch.sqrt(dx ** 2 + dy ** 2)

    # 取 yaw，默认是最后一个维度
    yaw1 = point_s[-1].item()
    yaw2 = curr[-1].item()
    dtheta = abs(angle_diff(yaw2, yaw1))

    # 判断可行性
    return dist <= max_dist and dtheta <= max_angle_diff



def RRT_points(point_s, point_e, max_dist=1.65, max_delta_theta=0.75, num_samples=1, rng=None, num_times = 100):

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
        if(is_valid_sample(samples, point_e, area, collision_area)):
            flag = True
    if(flag == False):
        return [-1, -1, -1]

    return samples


def revise_traj_fil(traj, max_samples=10):

    traj_satisfied = is_feasible(traj)
    time = 0
    while(not traj_satisfied and time <= max_samples):
        time += 1
        N = traj.shape[0]
        for i in range(N):
            curr = traj[i]
            point_s = traj[i - 1] if i > 0 else traj[i]  # 用前一帧作为参考（或当前）
            point_e = traj[i + 1] if i < N - 1 else traj[i]
            if(not is_feasible_point(point_s, curr)):
                sample = RRT_points(point_s=point_s, point_e=point_e, num_samples=1)
                if(sample[0] != -1):
                    traj[i] = sample

        traj_satisfied = is_feasible(traj)
    return traj

