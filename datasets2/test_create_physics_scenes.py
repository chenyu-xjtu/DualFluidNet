#!/usr/bin/env python3
"""此脚本使用SPlisHSPlasH生成随机流体序列"""
import os
import re
import argparse
from copy import deepcopy
import sys
import json
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial.transform import Rotation

from glob import glob
import time
import tempfile
import subprocess
from shutil import copyfile
import itertools

import open3d as o3d
from physics_data_helper import numpy_from_bgeo, write_bgeo_from_numpy
from splishsplash_config import SIMULATOR_BIN, VOLUME_SAMPLING_BIN

# 读取当前系统目录
SCRIPT_DIR = os.path.dirname(__file__)

# 用于在模拟中创建对象的一些常量
PARTICLE_RADIUS = 0.025
MAX_FLUID_START_VELOCITY_XZ = 2.0
MAX_FLUID_START_VELOCITY_Y = 0.5

MAX_RIGID_START_VELOCITY_XZ = 2.0
MAX_RIGID_START_VELOCITY_Y = 2.0

# 模拟的默认参数
default_configuration = {
    "pause": False,
    "stopAt": 16.0,
    "particleRadius": 0.025,
    "numberOfStepsPerRenderUpdate": 1,
    "density0": 1000,
    "simulationMethod": 4,
    "gravitation": [0, -9.81, 0],
    "cflMethod": 0,
    "cflFactor": 1,
    "cflMaxTimeStepSize": 0.005,
    "maxIterations": 100,
    "maxError": 0.01,
    "maxIterationsV": 100,
    "maxErrorV": 0.1,
    "stiffness": 50000,
    "exponent": 7,
    "velocityUpdateMethod": 0,
    "enableDivergenceSolver": True,
    "enablePartioExport": True,
    "enableRigidBodyExport": True,
    "particleFPS": 50.0,
    "partioAttributes": "density;velocity"
}

default_simulation = {
    "contactTolerance": 0.0125,
}

default_fluid = {
    "surfaceTension": 0.2,
    "surfaceTensionMethod": 0,
    "viscosity": 0.01,
    "viscosityMethod": 3,
    "viscoMaxIter": 200,
    "viscoMaxError": 0.05
}

default_rigidbody = {
    "translation": [0, 0, 0],
    "rotationAxis": [0, 1, 0],
    "rotationAngle": 0,
    "scale": [1.0, 1.0, 1.0],
    "color": [0.1, 0.4, 0.6, 1.0],
    "isDynamic": False,
    "isWall": True,
    "restitution": 0.6,
    "friction": 0.0,
    "collisionObjectType": 5,
    "collisionObjectScale": [1.0, 1.0, 1.0],
    "invertSDF": True,
}

default_fluidmodel = {"translation": [0.0, 0.0, 0.0], "scale": [1.0, 1.0, 1.0]}


def random_rotation_matrix(strength=None, dtype=None):
    """生成随机旋转矩阵

    strength: scalar in [0,1]. 1生成完全随机的旋转。0生成the identity(？)。默认值为1。
    dtype: 输出 dtype. 默认为 np.float32
    """
    if strength is None:
        strength = 1.0

    if dtype is None:
        dtype = np.float32

    x = np.random.rand(3)
    theta = x[0] * 2 * np.pi * strength
    phi = x[1] * 2 * np.pi
    z = x[2] * strength

    r = np.sqrt(z)
    V = np.array([np.sin(phi) * r, np.cos(phi) * r, np.sqrt(2.0 - z)])

    st = np.sin(theta)
    ct = np.cos(theta)

    Rz = np.array([[ct, st, 0], [-st, ct, 0], [0, 0, 1]])

    rand_R = (np.outer(V, V) - np.eye(3)).dot(Rz)
    return rand_R.astype(dtype)


def obj_volume_to_particles(objpath, scale=1, radius=None):
    """用于将物体（objpath指定的3D模型文件）体积转换为粒子

    objpath：表示输入的3D模型文件（物体）路径。
    scale：表示缩放因子，默认值为1。可以用于缩放模型的尺寸。
    radius：表示粒子的半径，默认为None。如果未指定，将使用全局变量PARTICLE_RADIUS的值作为粒子半径。
    """
    if radius is None:
        radius = PARTICLE_RADIUS
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = os.path.join(tmpdir, 'out.bgeo')
        scale_str = '{0}'.format(scale)
        radius_str = str(radius)

        # 将物体的体积转换为粒子，并将结果保存到指定的输出文件
        status = subprocess.run([
            VOLUME_SAMPLING_BIN, '-i', objpath, '-o', outpath, '-r', radius_str,
            '-s', scale_str
        ])
        return numpy_from_bgeo(outpath)  # 将outpath中的数据转换为NumPy数组


def obj_surface_to_particles(objpath, radius=None):
    """将边界框转换为粒子，使用了三角面片法

    return： points:粒子位置信息  normals:法线信息
    """
    if radius is None:
        radius = PARTICLE_RADIUS
    obj = o3d.io.read_triangle_mesh(objpath)
    particle_area = np.pi * radius ** 2
    # 1.9以大致匹配SPlisHSPlasHs表面采样的点数
    num_points = int(1.9 * obj.get_surface_area() / particle_area)
    pcd = obj.sample_points_poisson_disk(num_points, use_triangle_normal=True)
    points = np.asarray(pcd.points).astype(np.float32)
    normals = -np.asarray(pcd.normals).astype(np.float32)
    return points, normals


def rasterize_points(points, voxel_size, particle_radius):
    """接受点云数据，将其栅格化，并返回栅格化后的网格空间信息和栅格化后的数据"""
    if not (voxel_size > 2 * particle_radius):
        raise ValueError(
            "voxel_size > 2*particle_radius is not true. {} > 2*{}".format(
                voxel_size, particle_radius))

    points_min = (points - particle_radius).min(axis=0)
    points_max = (points + particle_radius).max(axis=0)

    arr_min = np.floor_divide(points_min, voxel_size).astype(np.int32)
    arr_max = np.floor_divide(points_max, voxel_size).astype(np.int32) + 1

    arr_size = arr_max - arr_min

    arr = np.zeros(arr_size)

    offsets = []  # 计算偏移
    for z in range(-1, 2, 2):
        for y in range(-1, 2, 2):
            for x in range(-1, 2, 2):
                offsets.append(
                    np.array([
                        z * particle_radius, y * particle_radius,
                        x * particle_radius
                    ]))

    for offset in offsets:
        idx = np.floor_divide(points + offset, voxel_size).astype(
            np.int32) - arr_min
        arr[idx[:, 0], idx[:, 1], idx[:, 2]] = 1

    return arr_min, voxel_size, arr


def find_valid_fluid_start_positions(box_rasterized, fluid_rasterized):
    """在光栅化的自由空间和流体数据中找到合适的位置，以便在该位置放置流体，并进行相应的边界框更新。
    return 起始坐标
    """
    fluid_shape = np.array(fluid_rasterized[2].shape)
    box_shape = np.array(box_rasterized[2].shape)
    last_pos = box_shape - fluid_shape  # last_pos表示可以放置流体的最后一个起始位置。这是通过从箱子的形状中减去流体的形状来实现的。这样做确保了流体在任何一个轴向上都不会超出箱子的边界。

    valid_fluid_start_positions_arr = np.zeros(box_shape)
    for idx in itertools.product(range(0, last_pos[0] + 1),
                                 range(0, last_pos[1] + 1),
                                 range(0, last_pos[2] + 1)):
        # 使用 itertools.product 生成可能的起始位置坐标，并检查每个位置是否适合放置流体。
        pos = np.array(idx, np.int32)
        pos2 = pos + fluid_shape
        view = box_rasterized[2][pos[0]:pos2[0], pos[1]:pos2[1], pos[2]:pos2[2]]
        if np.all(
                np.logical_and(view, fluid_rasterized[2]) ==
                fluid_rasterized[2]):
            if idx[1] == 0:
                valid_fluid_start_positions_arr[idx[0], idx[1], idx[2]] = 1
            elif np.count_nonzero(valid_fluid_start_positions_arr[idx[0],
                                  0:idx[1],
                                  idx[2]]) == 0:
                valid_fluid_start_positions_arr[idx[0], idx[1], idx[2]] = 1
        # 对于每个可能的起始位置，通过切片操作提取与流体体积相对应的箱子体积部分，检查这部分空间是否完全可用于放置流体。

    valid_pos = np.stack(np.nonzero(valid_fluid_start_positions_arr), axis=-1)  # 如果检测到合适的位置，将该位置标记为有效。
    selected_pos = valid_pos[np.random.randint(0, valid_pos.shape[0])]  # 在所有有效的起始位置中随机选择一个。

    # 通过减去流体体积来更新光栅化的边界框体积
    pos = selected_pos
    pos2 = pos + fluid_shape
    view = box_rasterized[2][pos[0]:pos2[0], pos[1]:pos2[1], pos[2]:pos2[2]]
    box_rasterized[2][pos[0]:pos2[0], pos[1]:pos2[1],
    pos[2]:pos2[2]] = np.logical_and(
        np.logical_not(fluid_rasterized[2]), view)

    selected_pos += box_rasterized[0]
    selected_pos = selected_pos.astype(np.float) * box_rasterized[1]

    # 返回计算出的流体应该放置的实际坐标位置。
    return selected_pos


def run_simulator(scene, output_dir):
    """为指定的场景文件运行模拟器"""
    with tempfile.TemporaryDirectory() as tmpdir:
        status = subprocess.run([
            SIMULATOR_BIN, '--no-cache', '--no-gui', '--no-initial-pause',
            '--output-dir', output_dir, scene
        ])


def create_fluid_data(output_dir, seed, options):
    """为特定种子创建随机场景并运行模拟器"""

    np.random.seed(seed)

    # 得到所有box*.obj文件的路径及文件名
    bounding_boxes = sorted(glob(os.path.join(SCRIPT_DIR, 'models',
                                              'Box*.obj')))
    # override bounding boxes 如果默认即使用box.obj
    if options.default_box:
        bounding_boxes = [os.path.join(SCRIPT_DIR, 'models', 'Box.obj')]

    # 得到所有Fluid*.obj文件的路径及文件名
    fluid_shapes = sorted(glob(os.path.join(SCRIPT_DIR, 'models',
                                            'Fluid*.obj')))
    rigid_shapes = sorted(
        glob(os.path.join(SCRIPT_DIR, 'models', 'RigidBody*.obj')))

    # 随机1 2 3
    num_objects = np.random.choice([1, 2, 3])  # 流体物体的数量1或2或3
    # override the number of objects to generate 如果设置了num_objects就改为该值
    if options.num_objects > 0:
        num_objects = options.num_objects
    # num_objects = random.choice([1])
    print('num_objects', num_objects)

    def create_fluid_object():
        """create fluids and place them randomly 创建流体并随机放置
        """
        fluid_obj = np.random.choice(fluid_shapes)  # 随机选择一个流体文件
        fluid = obj_volume_to_particles(fluid_obj,  # 将物体（objpath指定的3D模型文件）体积转换为粒子
                                        scale=np.random.uniform(0.5, 1.5))[0]
        R = random_rotation_matrix(1.0)  # 生成一个随机旋转矩阵
        fluid = fluid @ R  # 将之前转换后的粒子数据fluid与随机生成的旋转矩阵R相乘，实现了对粒子数据的随机旋转

        fluid_rasterized = rasterize_points(fluid, 2.01 * PARTICLE_RADIUS,  # 栅格化
                                            PARTICLE_RADIUS)

        # 在光栅化的自由空间中放置流体，找到有效的起始位置，并将流体放置在该位置上
        selected_pos = find_valid_fluid_start_positions(bb_rasterized,
                                                        fluid_rasterized)
        fluid_pos = selected_pos - fluid_rasterized[0] * fluid_rasterized[1]
        fluid += fluid_pos

        # 初始化流体的速度（fluid_vel），在三个方向上随机生成速度值来模拟流体的起始运动状态
        fluid_vel = np.zeros_like(fluid)
        max_vel = MAX_FLUID_START_VELOCITY_XZ
        fluid_vel[:, 0] = np.random.uniform(-max_vel, max_vel)
        fluid_vel[:, 2] = np.random.uniform(-max_vel, max_vel)
        max_vel = MAX_FLUID_START_VELOCITY_Y
        fluid_vel[:, 1] = np.random.uniform(-max_vel, max_vel)

        # 随机生成流体的密度和粘度，并根据不同选项对粘度进行额外的随机初始化
        density = np.random.uniform(500, 2000)
        viscosity = np.random.exponential(scale=1 / 20) + 0.01
        if options.uniform_viscosity:
            viscosity = np.random.uniform(0.01, 0.3)
        elif options.log10_uniform_viscosity:
            viscosity = 0.01 * 10 ** np.random.uniform(0.0, 1.5)

        if options.default_density:
            density = 1000
        if options.default_viscosity:
            viscosity = 0.01

        return {
            'type': 'fluid',
            'positions': fluid,
            'velocities': fluid_vel,
            'density': density,
            'viscosity': viscosity,
        }

    scene_is_valid = False

    for create_scene_i in range(100):
        if scene_is_valid:
            break

        # select random bounding box 选择随机边界框
        bb_obj = np.random.choice(bounding_boxes)

        # convert bounding box to particles 将边界框转换为粒子
        bb, bb_normals = obj_surface_to_particles(bb_obj)
        # 将物体（objpath指定的3D模型文件）体积转换为粒子
        bb_vol = obj_volume_to_particles(bb_obj)[0]

        # rasterize free volume 光栅化自由体积
        bb_rasterized = rasterize_points(np.concatenate([bb_vol, bb], axis=0),
                                         2.01 * PARTICLE_RADIUS,
                                         PARTICLE_RADIUS)
        bb_rasterized = bb_rasterized[0], bb_rasterized[1], binary_erosion(  # 腐蚀/ 侵蚀操作
            bb_rasterized[2], structure=np.ones((3, 3, 3)), iterations=3)

        objects = []

        create_fn_list = [create_fluid_object]

        # 尝试为场景中的每个对象调用创建函数，并通过多次尝试来确保成功创建对象
        for object_i in range(num_objects):

            create_fn = np.random.choice(create_fn_list)

            create_success = False
            for i in range(10):
                if create_success:
                    break
                try:
                    obj = create_fn()
                    objects.append(obj)
                    create_success = True
                    print('create object success')
                except:
                    print('create object failed')
                    pass

        scene_is_valid = True

        def get_total_number_of_fluid_particles():
            """计算场景中所有流体对象的粒子总数"""
            num_particles = 0
            for obj in objects:
                if obj['type'] == 'fluid':
                    num_particles += obj['positions'].shape[0]
            return num_particles

        def get_smallest_fluid_object():
            """找到场景中粒子数最小的流体对象"""
            num_particles = 100000000
            obj_idx = -1
            for idx, obj in enumerate(objects):
                if obj['type'] == 'fluid':
                    if obj['positions'].shape[0] < num_particles:
                        obj_idx = idx
                    num_particles = min(obj['positions'].shape[0],
                                        num_particles)
            return obj_idx, num_particles

        total_number_of_fluid_particles = get_total_number_of_fluid_particles()

        # 根据设定的流体粒子数量限制，调整生成的流体对象和粒子数量
        if options.const_fluid_particles:
            if options.const_fluid_particles > total_number_of_fluid_particles:
                scene_is_valid = False
            else:
                while get_total_number_of_fluid_particles(
                ) != options.const_fluid_particles:
                    difference = get_total_number_of_fluid_particles(
                    ) - options.const_fluid_particles
                    obj_idx, num_particles = get_smallest_fluid_object()
                    if num_particles < difference:
                        del objects[obj_idx]
                    else:
                        objects[obj_idx]['positions'] = objects[obj_idx][
                                                            'positions'][:-difference]
                        objects[obj_idx]['velocities'] = objects[obj_idx][
                                                             'velocities'][:-difference]

        if options.max_fluid_particles:
            if options.max_fluid_particles < total_number_of_fluid_particles:
                scene_is_valid = False

    sim_directory = os.path.join(output_dir, 'sim_{0:04d}'.format(seed))
    os.makedirs(sim_directory, exist_ok=False)

    # generate scene json file 生成场景json文件
    scene = {
        'Configuration': default_configuration,
        'Simulation': default_simulation,
        # 'Fluid': default_fluid,
        'RigidBodies': [],
        'FluidModels': [],
    }
    rigid_body_next_id = 1

    # bounding box 创建了一个包含刚体相关信息的场景字典，并将刚体对象添加到场景的 RigidBodies 列表中，为模拟提供了刚体的初始设置。
    box_output_path = os.path.join(sim_directory, 'box.bgeo')
    write_bgeo_from_numpy(box_output_path, bb, bb_normals)

    box_obj_output_path = os.path.join(sim_directory, 'box.obj')
    copyfile(bb_obj, box_obj_output_path)

    rigid_body = deepcopy(default_rigidbody)
    rigid_body['id'] = rigid_body_next_id
    rigid_body_next_id += 1
    rigid_body['geometryFile'] = os.path.basename(
        os.path.abspath(box_obj_output_path))
    rigid_body['resolutionSDF'] = [64, 64, 64]
    rigid_body["collisionObjectType"] = 5
    scene['RigidBodies'].append(rigid_body)

    # 为每个流体对象生成相应的模拟数据和设置，然后将整个场景保存为 JSON 文件，并通过模拟器运行模拟，生成模拟数据。
    fluid_count = 0
    for obj in objects:
        fluid_id = 'fluid{0}'.format(fluid_count)
        fluid_count += 1
        fluid = deepcopy(default_fluid)
        fluid['viscosity'] = obj['viscosity']
        fluid['density0'] = obj['density']
        scene[fluid_id] = fluid

        fluid_model = deepcopy(default_fluidmodel)
        fluid_model['id'] = fluid_id

        fluid_output_path = os.path.join(sim_directory, fluid_id + '.bgeo')
        write_bgeo_from_numpy(fluid_output_path, obj['positions'],
                              obj['velocities'])
        fluid_model['particleFile'] = os.path.basename(fluid_output_path)
        scene['FluidModels'].append(fluid_model)

    scene_output_path = os.path.join(sim_directory, 'scene.json')
    with open(scene_output_path, 'w') as f:
        json.dump(scene, f, indent=4)

    run_simulator(os.path.abspath(scene_output_path), sim_directory)


def main():
    parser = argparse.ArgumentParser(description="创建物理模拟数据")
    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="输出目录")
    parser.add_argument("--seed",
                        type=int,
                        required=True,
                        help="初始化的随机种子")
    parser.add_argument(
        "--uniform-viscosity",
        action='store_true',
        help="从均匀分布中生成随机粘度值")
    parser.add_argument(
        "--log10-uniform-viscosity",
        action='store_true',
        help=
        "从log10空间中的均匀分布生成随机粘度值"
    )
    parser.add_argument(
        "--default-viscosity",
        action='store_true',
        help="强制所有生成的流体具有默认粘度")
    parser.add_argument(
        "--default-density",
        action='store_true',
        help="强制所有生成的对象具有默认密度")
    parser.add_argument(
        "--default-box",
        action='store_true',
        help="强制所有生成的场景使用默认边界框")
    parser.add_argument(
        "--num-objects",
        type=int,
        default=0,
        help=
        "要放置在场景中的对象数。0（默认值）表示从1到3的随机选择"
    )
    parser.add_argument(
        "--const-fluid-particles",
        type=int,
        default=0,
        help="如果设置，将生成恒定数量的粒子。")
    parser.add_argument("--max-fluid-particles",
                        type=int,
                        default=0,
                        help="如果设置，粒子数量将受到限制。")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)  # exist_ok 表示允许该目录已经存在

    create_fluid_data(args.output, args.seed, args)


if __name__ == '__main__':
    sys.exit(main())
