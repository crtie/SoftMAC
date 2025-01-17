# SoftMAC
**[Note]**: This code repository will undergo a code refactoring in December, 2023. 

Implemetation for our paper "SoftMAC: Differentiable Soft Body Simulation with Forecast-based Contact Model and Two-way Coupling with Articulated Rigid Bodies and Clothes" (in submission).

[[project website]](https://sites.google.com/view/softmac) [[paper link]](https://arxiv.org/abs/2312.03297)

## Installation
We provide soft-rigid and soft-cloth coupling separately. You can install them following the instructions in `soft_rigid/README.md` and `soft_cloth/README.md`.

(The underlying simulator for clothes is not open-source yet. We will update the links later.)

## Demos
We optimize the action sequences with SoftMAC for each task.

**Demo 1: Pour wine**
```bash
# run under ./soft_rigid
python3 demo_pour.py
```
![Pour wine gif](gifs/pour_merged.gif)

**Demo 2: Squeeze plasticine**
```bash
# run under ./soft_rigid
python3 demo_grip.py
```
![Squeeze plasticine gif](gifs/grip_merged.gif)

**Demo 3: Pull door**
```bash
# run under ./soft_rigid
python3 demo_door.py
```
![Pull door gif](gifs/door_merged.gif)

**Demo 4: Make taco**
```bash
# run under ./soft_cloth
python3 demo_taco.py
```
![Make taco gif](gifs/taco_merged.gif)

**Demo 5: Push towel**
```bash
# run under ./soft_cloth
python3 demo_hit.py
```
![Make taco gif](gifs/hit_saved_merged.gif)


## Note
Feel free to contact me at minliu2@cs.cmu.edu or create a Github issue if you have questions regarding setting up the repository, running examples or adding new examples.

## Citation
If you find our work useful, please consider citing:
```bash
@article{liu2023softmac,
  title={SoftMAC: Differentiable Soft Body Simulation with Forecast-based Contact Model and Two-way Coupling with Articulated Rigid Bodies and Clothes},
  author={Liu, Min and Yang, Gang and Luo, Siyuan and Yu, Chen and Shao, Lin},
  journal={arXiv preprint arXiv:2312.03297},
  year={2023}
}
```
