#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import re
from glob import glob
import time
import importlib
import json
import dataflow
import zstandard as zstd
import msgpack
import msgpack_numpy
msgpack_numpy.patch()
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
from physics_data_helper import numpy_from_bgeo, write_bgeo_from_numpy
from create_physics_scenes import obj_surface_to_particles, obj_volume_to_particles
import open3d as o3d

dataset_dir = "../DeepLagrangianFluids-master/datasets/ours_default_data"
class PhysicsSimDataFlow(dataflow.RNGDataFlow):
    """Data flow for msgpacks generated from SplishSplash simulations.
    """

    def __init__(self, files, random_rotation=False, shuffle=False, window=2):
        if not len(files):
            raise Exception("List of files must not be empty")
        if window < 1:
            raise Exception("window must be >=1 but is {}".format(window))
        self.files = files
        self.random_rotation = random_rotation
        self.shuffle = shuffle
        self.window = window

    def __iter__(self):
        decompressor = zstd.ZstdDecompressor()
        files_idxs = np.arange(len(self.files)) # files中train3200个.msgpack.zst文件, test320个.msgpack.zst文件。
        if self.shuffle:
            self.rng.shuffle(files_idxs) # 打乱帧的顺序

        for file_i in files_idxs: # 对每一个.msgpack.zst文件进行遍历
            # read all data from file
            with open(self.files[file_i], 'rb') as f:
                data = msgpack.unpackb(decompressor.decompress(f.read()),
                                       raw=False) #一个.msgpack.zst中的数据：50帧的流体粒子数据fram_id, scene_id, pos, vel, m, viscosity，第一帧还包含box数据box, box_normal

            data_idxs = np.arange(len(data) - self.window + 1) #windows=3, data_idxs:0-48
            if self.shuffle:
                self.rng.shuffle(data_idxs)

            # get box from first item. The box is valid for the whole file
            box = data[0]['box']
            box_normals = data[0]['box_normals']

            for data_i in data_idxs:
            #对当前的.msgpack.zst里的每一帧进行遍历
                if self.random_rotation:
                    angle_rad = self.rng.uniform(0, 2 * np.pi)
                    s = np.sin(angle_rad)
                    c = np.cos(angle_rad)
                    rand_R = np.array([c, 0, s, 0, 1, 0, -s, 0, c],
                                      dtype=np.float32).reshape((3, 3))

                if self.random_rotation:
                    sample = {
                        'box': np.matmul(box, rand_R),
                        'box_normals': np.matmul(box_normals, rand_R)
                    } # 给每一帧的盒子施加随机旋转（一帧一次）
                else:
                    sample = {'box': box, 'box_normals': box_normals}

                for time_i in range(self.window): #为每一帧数据加上后两帧数据（包含三帧的数据）

                    item = data[data_i + time_i]

                    for k in ('pos', 'vel'):
                        if self.random_rotation:
                            sample[k + str(time_i)] = np.matmul(item[k], rand_R) # 给每一帧的粒子施加随机旋转（一个帧一次） 【给粒子和盒子同时施加旋转相当于把整个场景都旋转(即旋转重力)】
                        else:
                            sample[k + str(time_i)] = item[k]

                    for k in ('m', 'viscosity', 'frame_id', 'scene_id'):
                        sample[k + str(time_i)] = item[k]

                yield sample #(最终每一帧有3帧的数据（pos,vel,m,viscosity,frame_id,scene_id）以及box和box_normal，共20个键值对)

def write_particles(path_without_ext, pos, vel=None, options=None):
    """Writes the particles as point cloud ply.
    Optionally writes particles as bgeo which also supports velocities.
    """
    arrs = {'pos': pos}
    if not vel is None:
        arrs['vel'] = vel
    np.savez(path_without_ext + '.npz', **arrs)

    if options and options.write_ply:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pos))
        o3d.io.write_point_cloud(path_without_ext + '.ply', pcd)

    if options and options.write_bgeo:
        write_bgeo_from_numpy(path_without_ext + '.bgeo', pos, vel)

def run_sim_torch(trainscript_module, weights_path, scene, num_steps,
                  output_dir, options, frame):
    import torch
    device = torch.device(options.device)

    # init the network
    model = trainscript_module.create_model()
    weights = torch.load(weights_path)
    # print(weights.get("model"))
    # model.load_state_dict(weights.get("model")) #如果用的是pretrained_model就不用.get("model")
    model.load_state_dict(weights.get("model")) #如果是自己训练的模型就需要.get("model")
    model.to(device)
    model.requires_grad_(False)

    # prepare static particles
    walls = []

    for x in scene['walls']:
        points = frame['box'][0].copy()
        normals = frame['box_normals'][0].copy()
        if 'invert_normals' in x and x['invert_normals']:
            normals = -normals
        points += np.asarray([x['translation']], dtype=np.float32)
        walls.append((points, normals))
    box = np.concatenate([x[0] for x in walls], axis=0)
    box_normals = np.concatenate([x[1] for x in walls], axis=0)

    # export static particles
    write_particles(os.path.join(output_dir, 'box'), box, box_normals, options)

    # compute lowest point for removing out of bounds particles
    # min_y = np.min(box[:, 1]) - 0.05 * (np.max(box[:, 1]) - np.min(box[:, 1]))
    min_x = np.min(box[:, 0])
    max_x = np.max(box[:, 0])
    min_y = np.min(box[:, 1])
    max_y = np.max(box[:, 1])
    min_z = np.min(box[:, 2])
    max_z = np.max(box[:, 2])
    box = torch.from_numpy(box).to(device)
    box_normals = torch.from_numpy(box_normals).to(device)

    # prepare fluids

    # sim_208粒子初速度
    # 创建一个存储 0 到 1727 点的坐标的 PyTorch 张量
    coords1 = torch.tensor([[-1.76116, 0.3223207, -0.3251787] for _ in range(1728)])
    # 创建一个存储 1728 到 2794 点的坐标的 PyTorch 张量
    coords2 = torch.tensor([[-1.3466647, 0.43700173, 0.42514825] for _ in range(1067)])
    # 将这两个张量连接起来以得到完整的坐标
    coords = torch.cat((coords1, coords2), dim=0)
    # 将 PyTorch 张量转换为 NumPy 数组
    velocities = coords.numpy()

    fluids = []
    for x in scene['fluids']:
        points = frame['pos0'][0].copy()
        points += np.asarray([x['translation']], dtype=np.float32)
        # velocities = np.empty_like(points)
        # velocities[:, 0] = x['velocity'][0]
        # velocities[:, 1] = x['velocity'][1]
        # velocities[:, 2] = x['velocity'][2]
        range_ = range(x['start'], x['stop'], x['step'])
        fluids.append(
            (points.astype(np.float32), velocities.astype(np.float32), range_))

    pos = np.empty(shape=(0, 3), dtype=np.float32)
    vel = np.empty_like(pos)

    for step in range(num_steps):
        # add from fluids to pos vel arrays
        for points, velocities, range_ in fluids:
            if step in range_:  # check if we have to add the fluid at this point in time
                pos = np.concatenate([pos, points], axis=0)
                vel = np.concatenate([vel, velocities], axis=0)

        if pos.shape[0]:
            fluid_output_path = os.path.join(output_dir,
                                             'fluid_{0:04d}'.format(step))
            if isinstance(pos, np.ndarray):
                write_particles(fluid_output_path, pos, vel, options)
            else:
                write_particles(fluid_output_path, pos.numpy(), vel.numpy(),
                                options)

            inputs = (torch.from_numpy(pos).to(device),
                      torch.from_numpy(vel).to(device), None, box, box_normals)
            pos, vel = model(inputs)
            pos = pos.cpu().numpy()
            vel = vel.cpu().numpy()

        # remove out of bounds particles
        if step % 1 == 0:
            print(step, 'num particles', pos.shape[0])
            # mask = pos[:, 1] > min_y
            mask = np.logical_and.reduce((
                pos[:, 0] >= min_x,
                pos[:, 0] <= max_x,
                pos[:, 1] >= min_y,
                pos[:, 1] <= max_y,
                pos[:, 2] >= min_z,
                pos[:, 2] <= max_z
            ))
            if np.count_nonzero(mask) < pos.shape[0]:
                pos = pos[mask]
                vel = vel[mask]

def read_data_val(files, **kwargs):
    return read_data(files=files,
                     batch_size=1,
                     repeat=False,
                     shuffle_buffer=None,
                     num_workers=1,
                     **kwargs)

def read_data(files=None,
              batch_size=1,
              window=2,
              random_rotation=False,
              repeat=False,
              shuffle_buffer=None,
              num_workers=1,
              cache_data=False):
    print(files[0:20], '...' if len(files) > 20 else '')

    # caching makes only sense if the data is finite
    if cache_data:
        if repeat == True:
            raise Exception("repeat must be False if cache_data==True")
        if random_rotation == True:
            raise Exception("random_rotation must be False if cache_data==True")
        if num_workers != 1:
            raise Exception("num_workers must be 1 if cache_data==True")

    df = PhysicsSimDataFlow(
        files=files,
        random_rotation=random_rotation,
        shuffle=True if shuffle_buffer else False,
        window=window,
    )

    if repeat:
        df = dataflow.RepeatedData(df, -1)

    if shuffle_buffer:
        df = dataflow.LocallyShuffleData(df, shuffle_buffer)

    if num_workers > 1:
        df = dataflow.MultiProcessRunnerZMQ(df, num_proc=num_workers)

    df = dataflow.BatchData(df, batch_size=batch_size, use_list=True)

    if cache_data:
        df = dataflow.CacheData(df)

    df.reset_state()
    return df

def main():
    parser = argparse.ArgumentParser(
        description=
        "Runs a fluid network on the given scene and saves the particle positions as npz sequence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("trainscript",
                        type=str,
                        help="The python training script.")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help=
        "The path to the .h5 network weights file for tensorflow ot the .pt weights file for torch."
    )
    parser.add_argument("--num_steps",
                        type=int,
                        default=250,
                        help="The number of simulation steps. Default is 250.")
    parser.add_argument("--scene",
                        type=str,
                        required=True,
                        help="A json file which describes the scene.")
    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="The output directory for the particle data.")
    parser.add_argument("--write-ply",
                        action='store_true',
                        help="Export particle data also as .ply sequence")
    parser.add_argument("--write-bgeo",
                        action='store_true',
                        help="Export particle data also as .bgeo sequence")
    parser.add_argument("--device",
                        type=str,
                        default='cuda',
                        help="The device to use. Applies only for torch.")

    args = parser.parse_args()
    print(args)

    module_name = os.path.splitext(os.path.basename(args.trainscript))[0]
    sys.path.append('.')
    trainscript_module = importlib.import_module(module_name)

    val_files = sorted(glob(os.path.join(dataset_dir, 'valid', '*.zst')))
    val_dataset = read_data_val(files=val_files, window=1, cache_data=True)
    frame = None
    for data in val_dataset:
        if data['scene_id0'][0] == 'sim_0208' and data['frame_id0'][0] == 0:
            frame = data

    with open(args.scene, 'r') as f:
        scene = json.load(f)

    os.makedirs(args.output)
    if frame == None:
        print("No frame")
        return

    return run_sim_torch(trainscript_module, args.weights, scene,
                             args.num_steps, args.output, args, frame)


if __name__ == '__main__':
    sys.exit(main())
