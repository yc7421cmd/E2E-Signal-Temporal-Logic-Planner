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
from test import *
from robust import *
import math
import numpy as np


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


def is_in_collision_point(point, collision_area):
    """
    判断单个点是否落入任一 collision 区域。

    参数
    ----
    point : array-like of (x, y) or (x, y, heading)
    collision_area : list of [[x_min, x_max], [y_min, y_max]]

    返回
    ----
    bool : True 表示碰撞，False 表示未碰撞
    """
    x, y = point[0], point[1]
    for x_range, y_range in collision_area:
        if x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1]:
            return True
    return False


def RRT_feas_points(point_s, point_e, max_dist=1.65, max_delta_theta=0.75, num_samples=1, rng=None, num_times = 100):

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


def revise_traj(expr, traj, label, max_samples=10):

    robust = robust_cal([expr], torch.from_numpy(traj).unsqueeze(0))
    # print(robust)
    stl_satisfied = robust > 0
    in_area = is_within_area(traj)
    no_collision = not is_in_collision(traj, label)
    ok = stl_satisfied and in_area and no_collision
    if(ok):
        return traj
    time = 0
    while(not ok and time <= max_samples):

        """
        对非法轨迹点进行替换修复。

        参数
        ----
        traj : np.ndarray, shape (N, 3)
            原始轨迹，每行为 (x, y, heading)
        max_samples : int
            每个非法点尝试采样的次数上限

        返回
        ----
        new_traj : np.ndarray, shape (N, 3)
            修复后的轨迹（若无法修复某点，则保留原值）
        """
        new_traj = traj.copy()
        N = traj.shape[0]
        for i in range(N):
            cur_point = traj[i]
            if(is_in_collision_point(cur_point, collision_area)):
                print("now point is: ", i)
                point_s = traj[i - 1] if i > 0 else traj[i]  # 用前一帧作为参考（或当前）
                point_e = traj[i + 1] if i < N - 1 else traj[i]
                sample = RRT_feas_points(point_s=point_s, point_e=point_e, num_samples=1)
                if(sample[0] != -1):
                    new_traj[i] = sample

        robust = robust_cal([expr], torch.from_numpy(new_traj).unsqueeze(0))
        # print(robust)
        stl_satisfied = robust > 0
        in_area = is_within_area(new_traj)
        no_collision = not is_in_collision(new_traj, label)
        ok = stl_satisfied and in_area and no_collision

        time += 1

    if(not ok):
        return traj

    return new_traj



