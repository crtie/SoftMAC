import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np
import copy
import open3d as o3d
from pdb import set_trace as bp
from utils import fps_select
import time
import torch
from pointnet2_ops import pointnet2_utils

def fps_cuda(p,num_sample):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    p = torch.tensor(p, dtype=torch.float32).to(device)
    t1 = time.time()
    idx = pointnet2_utils.furthest_point_sample(p.unsqueeze(0).contiguous(), num_sample).squeeze()
    idx = idx.cpu().numpy()
    p = p[idx].cpu().numpy()
    p = p.astype(np.float64)
    return p
