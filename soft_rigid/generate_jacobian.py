import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import shutil
import time
from argparse import ArgumentParser
import pickle
import taichi as ti
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import copy
from matplotlib.ticker import ScalarFormatter
from pathlib import Path
from engine.taichi_env import TaichiEnv
import time
from tqdm import tqdm, trange
from utils import make_movie, render, prepare, fps_select, knn_select, K_Means, estimate_normals
from pdb import set_trace as bp
np.set_printoptions(precision=4)

class Controller:
    def __init__(
        self, steps=200, substeps=4000, n_controllers=1, actions_init=None,
        lr=1e-2, warmup=5, decay=1.0, betas=(0.9, 0.999),
    ):
        # actions
        self.steps = steps
        self.substeps = substeps
        self.n_controllers = n_controllers
        if actions_init is None:
            self.action = torch.zeros(steps, 3 * n_controllers, requires_grad=True)
        else:
            if actions_init.shape[0] > steps:
                assert actions_init.shape[0] == substeps
                actions_init = actions_init.reshape(steps, -1, 3 * n_controllers).mean(axis=1)
            self.action = actions_init.clone().detach().requires_grad_(True)
        # print("self.action",self.action.shape)

        # optimizer
        self.optimizer = optim.Adam([self.action, ], betas=betas)

        self.lr = lr
        self.decay = decay
        self.warmup = warmup

        # log
        self.epoch = 0

    def get_actions(self):
        return torch.tensor(self.action.detach().numpy().repeat(self.substeps // self.steps, axis=0))

    def schedule_lr(self):
        if self.epoch < self.warmup:
            lr = self.lr * self.epoch / self.warmup
        else:
            lr = self.lr * self.decay ** (self.epoch - self.warmup)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        self.latest_lr = lr

    def step(self, grad):
        self.schedule_lr()
        if grad.shape[0] > self.steps:
            grad = grad.reshape(self.steps, -1, 3 * self.n_controllers).mean(axis=1)
        # grad[:, 1] *= 1.0
        grad[:, 1] *= 0.
        actions_grad = grad

        self.action.backward(actions_grad)
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.epoch += 1

def get_init_actions(args, env, n_controllers=1, choice=0):
    if choice == 0:
        actions = torch.zeros(args.steps, n_controllers, 3)
    if choice == 1:
        actions = torch.zeros(args.steps,n_controllers, 3)
        # actions[:, 1] = 0.
    else:
        assert False
    return torch.FloatTensor(actions)

def plot_actions(log_dir, actions, actions_grad, epoch):
    actions = actions.detach().numpy()
    plt.figure()

    plt.subplot(211)
    plt.title("Actor 1")
    for i, axis in zip(range(3), ['x', 'y', 'z']):
        plt.plot(actions[:, i], label=axis)
    plt.legend(loc='upper right')

    plt.subplot(212)
    plt.title("Grad for Actor 1")
    for i, axis in zip(range(3), ['x', 'y', 'z']):
        plt.plot(actions_grad[:, i], label=axis)
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(log_dir / "actions" / f"actions_{epoch}.png", dpi=300)
    plt.close()

    torch.save(actions, log_dir / "ckpt" / f"actions_{epoch}.pt")

def choose_control_idx(env, num_controller):
    control_idx = -torch.ones(env.simulator.n_particles)
    init_particles, _ = env.shapes.get()

    k_means = K_Means(n_clusters=num_controller)
    k_means.fit(init_particles)  #计算聚类中心
    cat = k_means.predict(init_particles) #确定了聚类中心以后，计算每个点属于哪个聚类中心
    return np.array(cat, dtype=np.int32)

def get_key_points(env, num_key_points):
    init_particles, _ = env.shapes.get()
    cluster_centers = fps_select(init_particles, num_key_points)
    return cluster_centers

def flatten_grad(actions_grad, num_controller, control_idx):
    num_particles = control_idx.shape[0]
    gradient = np.zeros((num_particles, 3))
    for i in range(num_controller):
        gradient[control_idx == i] = actions_grad[i]
    return gradient

def get_center_point(points):
    #输出一堆点中位于中心的一个点
    # points: (n_points, dim)
    # return (1,)
    center = np.mean(points, axis=0)
    n_points = points.shape[0]
    dist = np.linalg.norm(points - center, axis=1)
    selected_idx = np.argsort(dist)[0]
    return np.array(selected_idx)


# Add the following to "engine/renderer/renderer.py: def set_primitives(self, f)" for btter visualization
# mesh.visual.face_colors = np.array([
#     [0.6, 0.6, 0.68, 1.0] for i in range(12)
# ] + [
#     [0.1, 0.1, 0.1, 1.0] for i in range(36)
# ])

def main(args, pts_file=None):
    # Path and Configurations
    log_dir, cfg = prepare(args)
    ckpt_dir = log_dir / "ckpt"
    ckpt_dir.mkdir(exist_ok=True)

    category = pts_file.split('/')[-2]
    save_dir = Path(pts_file).parent.parent / "jacobian" / category
    print("save_dir",save_dir)
    object_name = Path(pts_file).stem
    print("object_name",object_name)
    if os.path.exists(f'{save_dir}/{object_name}.pkl'):
        print(f'{save_dir}/{object_name}.pkl exists, skip')
        return
    cfg.SHAPES[0]['path'] = Path(pts_file)
    # print("cfg.SHAPES[0]['path']",cfg.SHAPES[0]['path'])

    save_dir.mkdir(exist_ok=True, parents=True)
    saved_data_pieces = 0
    saved_data = []

    env = TaichiEnv(cfg, loss=True)
    env.set_control_mode("mpm")
    env.initialize()

    total_time = 0.
    for epoch in trange(args.epochs):
        tik = time.time()
        # Build Environment

        # env.rigid_simulator.ext_grad_scale = 1 / 40.        # it works, but don't know why...
        num_controller = cfg.SIMULATOR.n_controllers
        control_idx = choose_control_idx(env, num_controller)
        env.simulator.set_control_idx(control_idx)
        actions = get_init_actions(args, env, n_controllers=num_controller, choice=1)
        controller = Controller(
            steps=args.steps // 1, substeps=args.steps, actions_init=actions,
            lr=1e-1, warmup=5, decay=0.99, betas=(0.5, 0.999), n_controllers=num_controller,
        )

        init_state, _ = env.shapes.get()
        loss_log = []
        #! random choose a controller and execute a random action
        key_points = get_key_points(env, cfg.SIMULATOR.n_key_points)
        num_used_controller = np.random.randint(1,cfg.max_attached_points+1) #! random choose a number of controllers(1~4)
        chosen_controller = np.random.randint(num_controller, size=num_used_controller)
        actions_applied = torch.zeros(10, num_controller, 3)
        actions_applied[:, chosen_controller] = torch.rand(args.steps, num_used_controller, 3) - 0.5
        # print("chosen_controller",chosen_controller)
        # bp()
        for i in range(10):
            env.forward(actions_applied[i])
        target_state = env.simulator.get_state(10)[:,:3]
        key_motion = (target_state - init_state)[key_points]
        init_normal = estimate_normals(init_state)
        target_normal = estimate_normals(target_state)

        #! preparation
        E_candidate = np.linspace(10, 100, 10)
        nu_candidate = np.linspace(0.1, 0.3, 10)
        cfg.SIMULATOR.E = int(np.random.choice(E_candidate))
        cfg.SIMULATOR.nu = float(np.random.choice(nu_candidate))
        env.simulator.set_material(cfg.SIMULATOR.E, cfg.SIMULATOR.nu)
        # print("E",cfg.SIMULATOR.E,"nu",cfg.SIMULATOR.nu)
        ti.ad.clear_all_gradients()
        env.initialize()
        # print("lamda", env.simulator.lam, "mu", env.simulator.mu)
        # bp()
        init_particles, _ = env.shapes.get()
        num_particles = init_particles.shape[0]
        jacobian = np.zeros((cfg.SIMULATOR.n_key_points, num_particles, 3, 3))
        ##! jacobian is supposed to be (num_key_points, num_particles, 3, 3)
        prepare_time = time.time() - tik
        tik = time.time()
        #! compute jacobian
        for i in range(len(key_points)):
            for j in range(3):
                ti.ad.clear_all_gradients()
                env.initialize()
                # tik = time.time()
                actions = controller.get_actions()
                for k in range(args.steps):
                    env.forward(actions[k])
                # forward_time = time.time() - tik

                # loss
                # tik = time.time()
                loss, pose_loss, vel_loss, contact_loss = 0., 0., 0., 0.
                with ti.ad.Tape(loss=env.loss.loss):
                    # print("compute loss", f)
                    point_idx = int(key_points[i])
                    target_motion = np.float64(key_motion[i][j])
                    loss_info = env.compute_loss(f = 1, point_idx = point_idx, sub_idx = j, key_motion = target_motion)
                    loss = loss_info["loss"]

                actions_grad = env.backward()
                # print("actions_grad",actions_grad.shape)
                # bp()
                jacobian_single_row = flatten_grad(actions_grad, num_controller, control_idx)
                jacobian[i,:,j,:] = jacobian_single_row
        jacobian_time = time.time() - tik
        # print("jacobian_time",jacobian_time)
        # applusible_attached_points = np.where(control_idx==chosen_controller)[0]
        # for row_idx in range(10):
        #     for potential_attached_point in applusible_attached_points:
        #         row_idx = np.random.randint(0, len(key_points))
        #         # col_idx = np.random.randint(0, num_particles)
        #         print("jacobian[{}, {}, :, :]".format(row_idx, potential_attached_point))
        #         print(jacobian[row_idx, potential_attached_point, :, :])
        #         bp()
        # render(env, log_dir, 0, n_steps=args.steps, interval=1, control_idx=control_idx)
        # make_movie(log_dir, f"epoch{epoch}")
        # bp()
        #! save data
        # tik = time.time()
        data = {}
        data['init_state'] = init_state
        data['target_state'] = target_state
        data['keypoints'] = key_points
        data['target_state_normal'] = target_normal
        data['init_state_normal'] = init_normal
        data['response_matrix'] = jacobian
        data['action'] = np.zeros((cfg.max_attached_points, 3))
        data['action'][:num_used_controller] = actions_applied[0, chosen_controller].detach().numpy()
        data['num_attached_points'] = num_used_controller
        data['E'] = cfg.SIMULATOR.E
        data['nu'] = cfg.SIMULATOR.nu
        # print("saved_E",cfg.SIMULATOR.E,"saved_nu",cfg.SIMULATOR.nu)
        data['attached_point_target'] = np.zeros(cfg.max_attached_points)
        for att_pts in range(num_used_controller):
            applusible_attached_points = np.where(control_idx==chosen_controller[att_pts])[0]
            # print("applusible_attached_points",applusible_attached_points)
            data['attached_point_target'][att_pts] = applusible_attached_points[get_center_point(init_state[applusible_attached_points])]
            assert int(data['attached_point_target'][att_pts]) in applusible_attached_points
        # print("attached_point_target",data['attached_point_target'])
        # print("num_used_controller",num_used_controller)
        # print("action",data['action'])
        # bp()
        saved_data.append(copy.deepcopy(data))
        saved_data_pieces += 1
        # save_time = time.time() - tik
        epoch_time = prepare_time + jacobian_time #+ save_time
        total_time += epoch_time
        # print(f"-------------epoch {epoch}---------------")
        # print("epoch time {:.3f}".format(epoch_time), "prepare_time {:.3f}".format(prepare_time), \
        #     "jacobian_time {:.3f}".format(jacobian_time))
        # print("total_data ",saved_data_pieces, "total_time {:.3f}".format(total_time))
        if (epoch+1) % args.save_interval == 0 :
            tik = time.time()
            f = open(f'{save_dir}/{object_name}.pkl', 'wb+')
            pickle.dump(saved_data, f)
            f.close()
            saved_data_pieces += len(saved_data)
            save_time = time.time() - tik
            # print("save {} data, takes {:.3f} seconds".format(len(saved_data),save_time))
            saved_data = []


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp-name", "-n", type=str, default="door")
    parser.add_argument("--config", type=str, default="config/generate_jacobian_config.py")
    parser.add_argument("--render-interval", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--data_path", type=str, default="/data/partical_object")
    args = parser.parse_args()
    filePath = args.data_path
    categories = os.listdir(filePath)
    legal_categories = ['boat','car','chair','guitar','knife','motorcycle','pistol', 'plane', 'skateboard', 'train']
    # legal_categories = ['plane']
    object_list = []
    for category in categories:
        if category not in legal_categories:
            continue
        cur_path = os.path.join(filePath, category)
        objs = os.listdir(cur_path)
        for obj in objs:
            obj_path = os.path.join(cur_path, obj)
            if os.path.isdir(obj_path):
                continue
            if obj[-4:] != ".npy":
                # os.remove(obj_path)
                # print(f"remove {obj_path}")
                continue
            object_list.append(obj_path)
    for i in trange(len(object_list)):
        main(args,object_list[i])
        # bp()
