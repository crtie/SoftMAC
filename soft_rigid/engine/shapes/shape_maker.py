import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np
import copy
import open3d as o3d
from pdb import set_trace as bp
import time
import torch
from pointnet2_ops import pointnet2_utils
from .utils import fps_cuda
COLORS = [
    (127 << 16) + 127,
    (127 << 8),
    127,
    127 << 16,
]


class Shapes:
    # make shapes from the configuration
    def __init__(self, cfg):
        self.objects = []
        self.colors = []

        self.dim = 3

        state = np.random.get_state()
        np.random.seed(0) #fix seed 0
        for i in cfg:
            kwargs = {key: eval(val) if isinstance(val, str) else val for key, val in i.items() if key!='shape'}
            print(kwargs)
            if i['shape'] == 'box':
                self.add_box(**kwargs)
            elif i['shape'] == 'sphere':
                self.add_sphere(**kwargs)
            elif i['shape'] == 'predefined':
                self.add_predefined(**kwargs)
            # elif i['shape'] == 'mesh':
            #     self.add_mesh(**kwargs)
            else:
                raise NotImplementedError(f"Shape {i['shape']} is not supported!")
        np.random.set_state(state)

    def get_n_particles(self, volume):
        return max(int(volume/0.2**3) * 10000, 1)

    def add_object(self, particles, color=None, init_rot=None):
        if init_rot is not None:
            import transforms3d
            q = transforms3d.quaternions.quat2mat(init_rot)
            origin = particles.mean(axis=0)
            particles[:, :self.dim] = (particles[:, :self.dim] - origin) @ q.T + origin
        # self.objects.append(particles[:,:self.dim])
        self.objects.append(particles)
        if color is None or isinstance(color, int):
            tmp = COLORS[len(self.objects)-1] if color is None else color
            color = np.zeros(len(particles), np.int32)
            color[:] = tmp
        self.colors.append(color)

    def add_box(self, init_pos, width, n_particles=10000, color=None, init_rot=None):
        # pass
        if isinstance(width, float):
            width = np.array([width] * self.dim)
        else:
            width = np.array(width)
        if n_particles is None:
            n_particles = self.get_n_particles(np.prod(width))
        p = (np.random.random((n_particles, self.dim)) * 2 - 1) * (0.5 * width) + np.array(init_pos)
        print(type(p), p.shape, p.dtype)
        self.add_object(p, color, init_rot=init_rot)

    def add_sphere(self, init_pos, radius, n_particles=10000, color=None, init_rot=None):
        if n_particles is None:
            if self.dim == 3:
                volume = (radius ** 3) * 4 * np.pi / 3
            else:
                volume = (radius ** 2) * np.pi
            n_particles = self.get_n_particles(volume)

        p = np.random.normal(size=(n_particles, self.dim))
        p /= np.linalg.norm(p, axis=-1, keepdims=True)
        # u = np.random.random(size=(n_particles, 1)) ** (1. / self.dim)
        u = 1.
        p = p * u * radius + np.array(init_pos)[:self.dim]
        self.add_object(p, color, init_rot=init_rot)

    def add_predefined(self, path, offset=None, color=None, scale=1.):
        if offset is None:
            offset = np.zeros(self.dim)
        p = np.load(path)
        center = p.mean(axis=0)
        p_norm = p - center
        p_norm *= scale
        p = p_norm + center
        p[:, :self.dim] += offset

        # print(p[:50])
        print(type(p), p.shape, p.dtype)
        self.add_object(p, color)

    def get(self):
        assert len(self.objects) > 0, "please add at least one shape into the scene"
        return np.concatenate(self.objects), np.concatenate(self.colors)
