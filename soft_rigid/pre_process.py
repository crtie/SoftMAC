import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import shutil
import time
from argparse import ArgumentParser
import pickle
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import copy
import open3d as o3d
from matplotlib.ticker import ScalarFormatter

from utils import make_movie, render, prepare, fps_select,fps_cuda, knn_select, K_Means, estimate_normals
from pdb import set_trace as bp
np.set_printoptions(precision=4)

def main(args):
    filePath = args.data_path
    categories = os.listdir(filePath)
    legal_categories = ['boat','car','chair','guitar','knife','motorcycle','pistol', 'plane', 'skateboard', 'train']
    # legal_categories = ['boat']


    #! ------------count data----------------
    # count = {key:0 for key in legal_categories}
    # for category in categories:
    #     if category not in legal_categories:
    #         continue
    #     cur_path = os.path.join(filePath, category)
    #     objs = os.listdir(cur_path)
    #     for obj in objs:
    #         obj_path = os.path.join(cur_path, obj)
    #         if os.path.isdir(obj_path):
    #             continue
    #         if obj[-4:] != ".obj":
    #             continue
    #         count[category] += 1
    # print(count)


    #!-----------fps and save----------------
    for category in categories:
        if category not in legal_categories:
            continue
        cur_path = os.path.join(filePath, category)
        objs = os.listdir(cur_path)
        for obj in objs:
            obj_path = os.path.join(cur_path, obj)
            if os.path.isdir(obj_path):
                continue
            if obj[-4:] != ".obj":
                # os.remove(obj_path)
                # print(f"remove {obj_path}")
                continue
            obj_identify = obj[:-4]
            # if obj_identify + ".npy" in objs:
            #     continue
            t1 = time.time()
            mesh = o3d.io.read_triangle_mesh(obj_path)
            p = np.array(mesh.vertices)
            if p.shape[0] < args.sample_points:
                print(f"skip {obj_path}")
                continue
            p = fps_cuda(p, args.sample_points)
            #* normalize and rescale
            # center = p.mean(axis=0)
            # p_norm = p - center
            # scale = np.max(np.linalg.norm(p_norm, axis=1))
            # p_norm = p_norm / scale
            
            save_path = os.path.join(cur_path, obj_identify)
            print(f"fps {obj_path} time {time.time()-t1}")
            np.save(save_path, p)



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/data/partical_object")
    parser.add_argument("--sample_points", type=int, default=2048)
    args = parser.parse_args()
    main(args)
