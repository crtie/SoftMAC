import cv2
import os
import numpy as np 
from pathlib import Path
from config import load
import json

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
    distances[farthest] = 0
    for i in range(n_samples):
        farthest = np.argmax(distances)
        selected_idx.append(farthest)
        for j in range(n_points):
            distances[j] = min(distances[j], np.linalg.norm(points[j] - points[farthest]))
    return np.array(selected_idx)

def knn_select(points, center_idx ,n_samples):
    # points: (n_points, dim)
    # return: (n_samples, )
    center = points[center_idx]
    n_points = points.shape[0]
    dist = np.linalg.norm(points - center, axis=1)
    selected_idx = np.argsort(dist)[:n_samples]
    return np.array(selected_idx)