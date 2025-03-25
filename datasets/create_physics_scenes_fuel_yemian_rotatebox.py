#!/usr/bin/env python3
"""This script generates random fluid sequences with SPlisHSPlasH."""
import os
import re
import argparse
from copy import deepcopy
import sys
import json
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial.transform import Rotation
import math
from glob import glob
import time
import tempfile
import subprocess
from shutil import copyfile
import itertools

import open3d as o3d
from physics_data_helper import numpy_from_bgeo, write_bgeo_from_numpy
from splishsplash_config import SIMULATOR_BIN, VOLUME_SAMPLING_BIN
import torch
SCRIPT_DIR = os.path.dirname(__file__)

# some constants for creating the objects in the simulation
PARTICLE_RADIUS = 0.025
MAX_FLUID_START_VELOCITY_XZ = 2.0
MAX_FLUID_START_VELOCITY_Y = 0.5

MAX_RIGID_START_VELOCITY_XZ = 2.0
MAX_RIGID_START_VELOCITY_Y = 2.0

# default parameters for simulation
default_configuration = {
    "pause": False,
    "stopAt": 8.0, #stopAt * particleFPS = 总共生成的帧数
    "particleRadius": 0.025,
    "numberOfStepsPerRenderUpdate": 1,
    "density0": 782.885,
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

import numpy as np

def generate_random_force():
    # 随机生成外力，方向任意，合力大小范围在0到3G之间
    magnitude = torch.rand(1) * 3 * 9.81

    # 随机生成一个方向，使用球坐标系
    theta = torch.rand(1) * 2 * math.pi  # 0到2π之间
    phi = torch.rand(1) * math.pi  # 0到π之间

    # 将球坐标转换为笛卡尔坐标
    x = magnitude * torch.sin(phi) * torch.cos(theta)
    y = magnitude * torch.sin(phi) * torch.sin(theta)
    z = magnitude * torch.cos(phi)

    force_tensor = torch.tensor([x, y, z])
    return force_tensor.tolist()

def random_rotation_matrix(strength=None, dtype=None):
    """Generates a random rotation matrix for fully random rotations

    strength: scalar in [0,1]. 1 generates fully random rotations. 0 generates the identity. Default is 1.
    dtype: output dtype. Default is np.float32
    """
    if strength is None:
        strength = 1.0

    if dtype is None:
        dtype = np.float32

    # Random rotation angles
    theta = np.random.rand() * 2 * np.pi * strength  # Rotation around Z-axis
    phi = np.random.rand() * 2 * np.pi * strength  # Rotation around Y-axis
    psi = np.random.rand() * 2 * np.pi * strength  # Rotation around X-axis

    # Rotation matrix around Z-axis
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    # Rotation matrix around Y-axis
    Ry = np.array([
        [np.cos(phi), 0, np.sin(phi)],
        [0, 1, 0],
        [-np.sin(phi), 0, np.cos(phi)]
    ])

    # Rotation matrix around X-axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(psi), -np.sin(psi)],
        [0, np.sin(psi), np.cos(psi)]
    ])

    # Combined rotation matrix: R = Rz * Ry * Rx
    rand_R = Rz.dot(Ry).dot(Rx)
    return rand_R.astype(dtype)

def obj_volume_to_particles(objpath, scale=1, radius=None):
    if radius is None:
        radius = PARTICLE_RADIUS
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = os.path.join(tmpdir, 'out.bgeo')
        scale_str = '{0}'.format(scale)
        radius_str = str(radius)
        status = subprocess.run([
            VOLUME_SAMPLING_BIN, '-i', objpath, '-o', outpath, '-r', radius_str,
            '-s', scale_str
        ])
        return numpy_from_bgeo(outpath)


def obj_surface_to_particles(objpath, radius=None):
    if radius is None:
        radius = PARTICLE_RADIUS
    obj = o3d.io.read_triangle_mesh(objpath)
    particle_area = np.pi * radius**2
    # 1.9 to roughly match the number of points of SPlisHSPlasHs surface sampling
    num_points = int(1.9 * obj.get_surface_area() / particle_area)
    pcd = obj.sample_points_poisson_disk(num_points, use_triangle_normal=True)
    points = np.asarray(pcd.points).astype(np.float32)
    normals = -np.asarray(pcd.normals).astype(np.float32)
    return points, normals

def run_simulator(scene, output_dir):
    """Runs the simulator for the specified scene file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        status = subprocess.run([
            SIMULATOR_BIN, '--no-cache', '--no-gui', '--no-initial-pause',
            '--output-dir', output_dir, scene
        ])

def apply_rotation(vertex, rotation_matrix):
    """Applies rotation to a vertex or normal."""
    v = np.array(vertex, dtype=np.float32)
    return rotation_matrix.dot(v)

def rotate_obj_file(input_filename, output_filename, rotation_matrix):
    """Rotate vertices and normals in an OBJ file and save to a new file."""
    with open(input_filename, 'r') as file:
        lines = file.readlines()

    with open(output_filename, 'w') as file:
        for line in lines:
            parts = line.strip().split()
            if line.startswith('v ') or line.startswith('vn '):  # Process vertices and vertex normals
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                rotated_vertex = apply_rotation(vertex, rotation_matrix)
                file.write(f"{parts[0]} {rotated_vertex[0]} {rotated_vertex[1]} {rotated_vertex[2]}\n")
            elif line.startswith('vt '):  # Process texture coordinates (no rotation)
                file.write(line)
            elif line.startswith('f '):  # Process faces (no change)
                file.write(line)
            else:
                # Write other lines as-is
                file.write(line)
    return output_filename

def create_fluid_data(output_dir, seed, options):
    """Creates a random scene for a specific seed and runs the simulator"""
    # 生成一个随机方向的力，力大小为0到3G
    # default_configuration['gravitation'] = generate_random_force()
    default_configuration['gravitation'] = [0, -9.81, 0]
    print("gravity", default_configuration['gravitation'])

    np.random.seed(seed)
    print("seed", seed)
    # 随机挑选一个盒子和一个液体
    # bounding_boxes = sorted(glob(os.path.join(SCRIPT_DIR, 'models',
    #                                           'Box*.obj')))
    # # override bounding boxes
    # if options.default_box:
    #     bounding_boxes = [os.path.join(SCRIPT_DIR, 'models', 'Box.obj')]
    # fluid_shapes = sorted(glob(os.path.join(SCRIPT_DIR, 'models', 'fueltank', 'big',
    #                                         'fluid_*.obj')))

    # 随机旋转
    # rand_R = random_rotation_matrix(1)

    # # 生成 -90 度到 90 度之间的随机角度(绕x轴转是横滚角，绕z轴转是俯仰角)
    random_angle_x = np.random.uniform(-90, 90)
    random_angle_z = np.random.uniform(-90, 90)
    # random_angle_x = -40
    # random_angle_z = -40
    print("油箱俯仰角（绕z轴）:", random_angle_z, "度")
    print("油箱横滚角（绕x轴）:", random_angle_x, "度")
    # random_angle_x = 0
    # random_angle_z = np.random.uniform(-10, 40)
    angle_x = math.radians(random_angle_x)
    angle_z = math.radians(random_angle_z)
    # 旋转矩阵
    rand_R_z = np.array([
        [math.cos(angle_z), -math.sin(angle_z), 0],
        [math.sin(angle_z), math.cos(angle_z), 0],
        [0, 0, 1]
    ]) #绕z轴
    rand_R_x = np.array([
        [1, 0, 0],
        [0, math.cos(angle_x), -math.sin(angle_x)],
        [0, math.sin(angle_x), math.cos(angle_x)]
    ]) #绕x轴
    rand_R = rand_R_x @ rand_R_z

    # gravity = np.array([0, -9.81, 0], dtype=np.float32)
    # new_gravity_direction = rand_R.T.dot(gravity)
    # print("现在的重力为" + str(new_gravity_direction))

    fluid_shapes = sorted(glob(os.path.join(SCRIPT_DIR, 'models', 'yemian', 'simple_tank', 'yemian', 's*.obj')))
    # fluid_shapes = sorted(glob(os.path.join(SCRIPT_DIR, 'models', 'triangle_fueltank', 'fluid', 'final', 'surface_180f0h.obj')))
    # 指定一个盒子和一个液体
    bounding_boxes = [os.path.join(SCRIPT_DIR, 'models', 'yemian', 'simple_tank', 'tank.obj')]
    # bounding_boxes = [os.path.join(SCRIPT_DIR, 'models', 'fueltank', 'tank111.obj')]

    rigid_shapes = sorted(
        glob(os.path.join(SCRIPT_DIR, 'models', 'RigidBody*.obj')))

    # files = os.listdir("/home/wny/code/dual_small/datasets/models/triangle_fueltank/fluid/final")   # 读入文件夹
    # num_objects = len(files)
    num_objects = 1
    # num_objects = np.random.choice([2, 3, 4]) # normaltank_big
    # num_objects = np.random.choice([4,5,6,7,8,9,10,11,12,13,14,15,16]) # normaltank_small
    # num_objects = np.random.choice(np.arange(25,31,1))  # triangle_fueltank

    # override the number of objects to generate
    # if options.num_objects > 0:
    #     num_objects = options.num_objects
    # num_objects = random.choice([1])
    print('num_objects', num_objects)


    def create_fluid_object():
    # create fluids and place them randomly
        fluid_obj = np.random.choice(fluid_shapes)
        print("流体块obj：", fluid_obj)
        fluid_shapes.remove(fluid_obj) # 为了液体不重复

        fluid_obj = rotate_obj_file(fluid_obj,
                                 options.rotated_obj_path+"/fluid/" + f"sim_{seed:04}_" + str(object_i) + '.obj',
                                 rand_R)

        fluid = obj_volume_to_particles(fluid_obj,
                                        scale=1)[0] #obj to particles

        fluid_vel = np.zeros_like(fluid)

        return {
            'type': 'fluid',
            'positions': fluid,
            'velocities': fluid_vel,
            'density': default_configuration['density0'],
            'viscosity': default_fluid['viscosity'],
        }

    scene_is_valid = False

    for create_scene_i in range(100):
        if scene_is_valid:
            break

        # select random bounding box
        bb_obj = np.random.choice(bounding_boxes)
        # 旋转盒子
        bb_obj = rotate_obj_file(bb_obj, options.rotated_obj_path+"/box/" +  f"sim_{seed:04}" + '_rotated_box.obj', rand_R)
        # convert bounding box to particles
        bb, bb_normals = obj_surface_to_particles(bb_obj)

        objects = []

        create_fn_list = [create_fluid_object] #create_fluid_object是create_fluid_data的嵌套函数，无需传参，可以直接使用外部函数create_fluid_data的参数

        for object_i in range(num_objects):

            create_fn = np.random.choice(create_fn_list)

            create_success = False
            for i in range(10): #十次随机生成的机会
                if create_success:
                    break
                try:
                    obj = create_fn()
                    objects.append(obj) #生成成功的液体加入objects中
                    create_success = True
                    print('create object success')
                except:
                    print('create object failed')
                    pass

        scene_is_valid = True

        def get_total_number_of_fluid_particles():
            num_particles = 0
            for obj in objects:
                if obj['type'] == 'fluid':
                    num_particles += obj['positions'].shape[0]
            return num_particles

        def get_smallest_fluid_object():
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

    # generate scene json file
    scene = {
        'Configuration': default_configuration,
        'Simulation': default_simulation,
        # 'Fluid': default_fluid,
        'RigidBodies': [],
        'FluidModels': [],
    }
    rigid_body_next_id = 1

    # bounding box
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
    parser = argparse.ArgumentParser(description="Creates physics sim data")
    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="The path to the output directory")
    parser.add_argument("--seed",
                        type=int,
                        required=True,
                        help="The random seed for initialization")
    parser.add_argument(
        "--uniform-viscosity",
        action='store_true',
        help="Generate a random viscosity value from a uniform distribution")
    parser.add_argument(
        "--log10-uniform-viscosity",
        action='store_true',
        help=
        "Generate a random viscosity value from a uniform distribution in log10 space"
    )
    parser.add_argument(
        "--default-viscosity",
        action='store_true',
        help="Force all generated fluids to have the default viscosity")
    parser.add_argument(
        "--default-density",
        action='store_true',
        default=782.885,
        help="Force all generated objects to have the default density")
    parser.add_argument(
        "--default-box",
        action='store_true',
        help="Force all generated scenes to use the default bounding box")
    parser.add_argument(
        "--num-objects",
        type=int,
        default=0,
        help=
        "The number of objects to place in the scene. 0 (default value) means random choice from 1 to 3"
    )
    parser.add_argument(
        "--const-fluid-particles",
        type=int,
        default=0,
        help="If set a constant number of particles will be generated.")
    parser.add_argument("--max-fluid-particles",
                        type=int,
                        default=0,
                        help="If set the number of particles will be limited.")
    parser.add_argument(
        "--rotated_obj_path",
        type=str,
        help="path to save rotated obj")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    create_fluid_data(args.output, args.seed, args)


if __name__ == '__main__':
    sys.exit(main())
