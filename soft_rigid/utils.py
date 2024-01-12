import cv2
import os
import numpy as np 
import random
from pathlib import Path
from config import load
import json
import open3d as o3d
# ===============================
# Rendering
# ===============================
def make_movie(log_dir, name=None):
    import imageio.v2 as imageio
    filenames = os.listdir(log_dir / "figs")
    filenames.sort()
    gif_name = "movie.gif" if name is None else name + ".gif"
    with imageio.get_writer(log_dir / gif_name, mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(log_dir / "figs" / filename)
            writer.append_data(image)
    print("Movie saved")
    os.system(f"rm -r {str(log_dir / 'figs')}")

def render(env, log_dir, epoch=0, action=None, n_steps=100, interval=10, control_idx=None):
    #! CHENRUI: control_idx is for rendering the attached points using different colors
    print("Rendering...")
    fig_dir = log_dir / "figs"
    fig_dir.mkdir(exist_ok=True)
    if action is not None:
        env.initialize()
        is_copy = env._is_copy
        env.set_copy(True)
    for i in range(n_steps):
        if action is not None:
            env.step(action[i])
        if i % interval == 0:
            frame = i * env.substeps if action is None else 0
            img = env.render(frame, control_idx=control_idx)
            img = img[:, :, ::-1]
            cv2.imwrite(str(fig_dir / f"{epoch:02d}-{i:05d}.png"), img)
            print(f"Frame {i} saved")
    if action is not None:
        env.set_copy(is_copy)


# ===============================
# Preparation
# ===============================
def prepare(args):
    Path("logs/").mkdir(exist_ok=True)
    log_dir = Path("logs/") / args.exp_name
    log_dir.mkdir(exist_ok=True)
    cfg = load(args.config)
    os.system(f"cp {args.config} {str(log_dir / 'config.py')}")
    with open(log_dir / "args.json", 'wt') as f:
        json.dump(vars(args), f, indent=4)

    fig_dir = log_dir / "figs"
    if os.path.exists(fig_dir):
        os.system(f"rm -r {str(fig_dir)}")
    fig_dir.mkdir(exist_ok=True)

    action_dir = log_dir / "actions"
    if os.path.exists(action_dir):
        os.system(f"rm -r {str(action_dir)}")
    action_dir.mkdir(exist_ok=True)
    return log_dir, cfg


# ===============================
# select control points
# ===============================
def fps_select(points, n_samples):
    # points: (n_points, dim)
    # return: (n_samples, )
    n_points = points.shape[0]
    selected_idx = []
    distances = np.zeros(n_points)
    distances[:] = 1e10
    farthest = np.random.randint(n_points)
    # print("first farthest",farthest)
    distances[farthest] = 0
    for i in range(n_samples):
        selected_idx.append(farthest)
        # print("farthest",farthest)
        for j in range(n_points):
            distances[j] = min(distances[j], np.linalg.norm(points[j] - points[farthest]))
        farthest = np.argmax(distances)
    return np.array(selected_idx)

from pointnet2_ops import pointnet2_utils
import torch

def fps_cuda(p,num_sample):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    p = torch.tensor(p, dtype=torch.float32).to(device)
    idx = pointnet2_utils.furthest_point_sample(p.unsqueeze(0).contiguous(), num_sample).squeeze()
    idx = idx.cpu().numpy()
    p = p[idx].cpu().numpy()
    p = p.astype(np.float64)
    return p


def knn_select(points, center_idx ,n_samples):
    # points: (n_points, dim)
    # return: (n_samples, )
    center = points[center_idx]
    n_points = points.shape[0]
    dist = np.linalg.norm(points - center, axis=1)
    selected_idx = np.argsort(dist)[:n_samples]
    return np.array(selected_idx)


# ===============================
# K_Means
# ===============================


class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
 
    #计算各个类别中心点的坐标
    def fit(self, data):
        # 随机选取数据中的中心点
        centers = data[random.sample(range(data.shape[0]), self.k_)]  #从 range(data.shape[0]) 数据中，随机抽取self.k_ 作为一个列表
        old_centers = np.copy(centers) #将旧的中心点 保存到old_centers
        labels = [ [] for i in range(self.k_) ]
        # [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]
        for iter_ in range(self.max_iter_):  # 循环一定的次数
            for idx, point in enumerate(data): # enumerate 函数用于一个可遍历的数据对象组合为一个索引序列，同时列出数据和数据的下标
                # 默认的二范数 就是距离  将每一个点计算到每个中心的距离  有2类，0 ，1 就是计算一点到2个中心点的距离  
                diff = np.linalg.norm(old_centers - point, axis=1)  # 一个点分别到两个中心点的距离不同，
                diff2 = (np.argmin(diff))  #np.argmin(diff) 表示最小值在数组中的位置  选取距离小的那一点的索引 也就代表了属于哪个类
                labels[diff2].append(idx) # 选取距离小的那一点的索引 也就代表了属于哪个类
 
            for i in range(self.k_):
                points = data[labels[i], :]   # 所有在第k类中的所有点
                centers[i] = points.mean(axis=0)  #均值 作为新的聚类中心
            if np.sum(np.abs(centers - old_centers)) < self.tolerance_ * self.k_:  #如果前后聚类中心的距离相差小于self.tolerance_ * self.k_ 输出
                break
            old_centers = np.copy(centers)
        self.centers = centers
        self.fitted = True
        # 屏蔽结束
 
    # 计算出各个点的类别
    def predict(self, p_datas):
        result = []
        # 作业2
        # 屏蔽开始
        if not self.fitted:
            print('Unfitter. ')
            return result
        for point in p_datas:
            diff = np.linalg.norm(self.centers - point, axis = 1)
            result.append(np.argmin(diff))
        # 屏蔽结束
        return result
 

# ===============================
# normal estimation
# ===============================
def estimate_normals(points, radius=0.01):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    normals = np.asarray(pcd.normals)
    return normals