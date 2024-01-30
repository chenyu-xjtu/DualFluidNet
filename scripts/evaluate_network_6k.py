#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np
import re
from glob import glob
import time
import importlib
import yaml
from pyemd import emd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datasets.dataset_reader_physics import read_data_val
from fluid_evaluation_helper import FluidErrors
from scipy.stats import wasserstein_distance
import torch
import torch.nn.functional as F
from open3d.ml.torch.python.layers.neighbor_search import FixedRadiusSearch

def window_poly6(r_sqr):
    return torch.clamp((1 - r_sqr) ** 3, 0, 1)  # torch.clamp()将输入input张量每个元素的范围限制到区间 [min,max]，返回结果到一个新张量。

def compute_density(out_pos, in_pos=None, radius=0.005, win=window_poly6):
    if in_pos is None:
        in_pos = out_pos

    if win is None:
        print("WARNING: No window function for density function!")
        win = lambda x: x

    radius = torch.tensor(radius)
    fixed_radius_search = FixedRadiusSearch()
    neighbors_index, neighbors_row_splits, dist = fixed_radius_search(
        in_pos, out_pos, radius)
    neighbors_index = neighbors_index.type(torch.int64).unsqueeze(1)
    neighbors = torch.gather(in_pos, 0, neighbors_index)
    dist = neighbors - out_pos.unsqueeze(1)
    dist = torch.sum(dist**2, dim=-1) / radius**2
    dens = torch.sum(win(dist), dim=-1)
    return dens

def density_loss(gt, pred, gt_in=None, pred_in=None, radius=0.005, eps=0.01, win=window_poly6, use_max=False, **kwargs):
    pred_dens = compute_density(pred, pred_in, radius, win=win)
    gt_dens = compute_density(gt, gt_in, radius, win=win)

    rest_dens = gt_dens.max()

    if use_max:
        return torch.abs(pred_dens.max() - rest_dens) / rest_dens

    err = F.relu(pred_dens - rest_dens - eps)
    return err.mean()

def rou_calculate(pos):
    rou = []
    filter_extent = np.float32(1.5 * 6 *
                               0.025)
    radius = 0.5 * torch.tensor(filter_extent)
    window_function = window_poly6
    fixed_radius_search = FixedRadiusSearch(
        metric='L2',
        ignore_query_point=False,
        return_distances=not window_function is None)
    neighbors_info1 = fixed_radius_search(torch.tensor(pos),
                                          queries=torch.tensor(pos), radius=0.5 * filter_extent,
                                          hash_table=None)
    neighbors_distance = neighbors_info1.neighbors_distance
    neighbors_index = neighbors_info1.neighbors_index
    neighbors_row_splits = neighbors_info1.neighbors_row_splits
    neighbors_distance_normalized = neighbors_distance / (radius * radius)
    neighbors_importance = window_function(neighbors_distance_normalized)

    for i in range(len(neighbors_row_splits) - 1):
        rou.append(0)
        for j in range(neighbors_row_splits[i], neighbors_row_splits[i + 1]):
            rou[i] += neighbors_importance[j]
    return rou
def evaluate_tf(model, val_dataset, frame_skip, fluid_errors=None, scale=1):
    print('evaluating.. ', end='')

    if fluid_errors is None:
        fluid_errors = FluidErrors()

    skip = frame_skip

    last_scene_id = 0
    frames = []
    for data in val_dataset:
        if data['frame_id0'][0] == 0:
            frames = []
        if data['frame_id0'][0] % skip < 3:
            frames.append(data)
        if data['frame_id0'][0] % skip == 3:

            if len(
                    set([
                        frames[0]['scene_id0'][0], frames[1]['scene_id0'][0],
                        frames[2]['scene_id0'][0]
                    ])) == 1:
                scene_id = frames[0]['scene_id0'][0]
                if last_scene_id != scene_id:
                    last_scene_id = scene_id
                    print(scene_id, end=' ', flush=True)
                frame0_id = frames[0]['frame_id0'][0]
                frame1_id = frames[1]['frame_id0'][0]
                frame2_id = frames[2]['frame_id0'][0]
                box = frames[0]['box'][0]
                box_normals = frames[0]['box_normals'][0]
                gt_pos1 = frames[1]['pos0'][0]
                gt_pos2 = frames[2]['pos0'][0]

                inputs = (frames[0]['pos0'][0], frames[0]['vel0'][0], None, box,
                          box_normals)
                pr_pos1, pr_vel1 = model(inputs)

                inputs = (pr_pos1, pr_vel1, None, box, box_normals)
                pr_pos2, pr_vel2 = model(inputs)

                fluid_errors.add_errors(scene_id, frame0_id, frame1_id,
                                        scale * pr_pos1, scale * gt_pos1)
                fluid_errors.add_errors(scene_id, frame0_id, frame2_id,
                                        scale * pr_pos2, scale * gt_pos2)

            frames = []

    result = {}
    result['err_n1'] = np.mean(
        [v['mean'] for k, v in fluid_errors.errors.items() if k[1] + 1 == k[2]])
    result['err_n2'] = np.mean(
        [v['mean'] for k, v in fluid_errors.errors.items() if k[1] + 2 == k[2]])

    print(result)
    print('done')

    return result


def evaluate_whole_sequence_tf(model,
                               val_dataset,
                               frame_skip,
                               fluid_errors=None,
                               scale=1):
    print('evaluating.. ', end='')

    if fluid_errors is None:
        fluid_errors = FluidErrors()

    skip = frame_skip

    last_scene_id = None
    for data in val_dataset:
        scene_id = data['scene_id0'][0]
        if last_scene_id is None or last_scene_id != scene_id:
            print(scene_id, end=' ', flush=True)
            last_scene_id = scene_id
            box = data['box'][0]
            box_normals = data['box_normals'][0]
            init_pos = data['pos0'][0]
            init_vel = data['vel0'][0]

            inputs = (init_pos, init_vel, None, box, box_normals)
        else:
            inputs = (pr_pos, pr_vel, None, box, box_normals)

        pr_pos, pr_vel = model(inputs)

        frame_id = data['frame_id0'][0]
        if frame_id > 0 and frame_id % skip == 0:
            gt_pos = data['pos0'][0]
            fluid_errors.add_errors(scene_id,
                                    0,
                                    frame_id,
                                    scale * pr_pos,
                                    scale * gt_pos,
                                    compute_gt2pred_distance=True)

    result = {}
    result['whole_seq_err'] = np.mean([
        v['gt2pred_mean']
        for k, v in fluid_errors.errors.items()
        if 'gt2pred_mean' in v
    ])

    print(result)
    print('done')

    return result


def evaluate_torch(model,
                   val_dataset,
                   frame_skip,
                   device,
                   fluid_errors=None,
                   scale=1):
    import torch
    print('evaluating.. ', end='')

    if fluid_errors is None:
        fluid_errors = FluidErrors()

    skip = frame_skip

    last_scene_id = 0
    frames = []
    emd_distance1 = []
    emd_distance2 = []
    rou_array = []
    for data in val_dataset:
        if data['frame_id0'][0] == 0:
            frames = []
        if data['frame_id0'][0] % skip < 3:
            frames.append(data)
        if data['frame_id0'][0] % skip == 3:
            #读3帧数据，后两帧是第一帧的gt
            if len(
                    set([
                        frames[0]['scene_id0'][0], frames[1]['scene_id0'][0],
                        frames[2]['scene_id0'][0]
                    ])) == 1:
                scene_id = frames[0]['scene_id0'][0]
                if last_scene_id != scene_id:
                    last_scene_id = scene_id
                    print(scene_id, end=' ', flush=True)
                frame0_id = frames[0]['frame_id0'][0]
                frame1_id = frames[1]['frame_id0'][0]
                frame2_id = frames[2]['frame_id0'][0]
                box = torch.from_numpy(frames[0]['box'][0]).to(device)
                box_normals = torch.from_numpy(
                    frames[0]['box_normals'][0]).to(device)
                gt_pos1 = frames[1]['pos0'][0]
                gt_pos2 = frames[2]['pos0'][0]

                inputs = (torch.from_numpy(frames[0]['pos0'][0]).to(device),
                          torch.from_numpy(frames[0]['vel0'][0]).to(device),
                          None, box, box_normals)
                pr_pos1, pr_vel1 = model(inputs)

                inputs = (pr_pos1, pr_vel1, None, box, box_normals)
                pr_pos2, pr_vel2 = model(inputs)

                #MSE
                fluid_errors.add_errors(scene_id, frame0_id, frame1_id,
                                        scale * pr_pos1.cpu().detach().numpy(),
                                        scale * gt_pos1) #第一帧误差
                fluid_errors.add_errors(scene_id, frame0_id, frame2_id,
                                        scale * pr_pos2.cpu().detach().numpy(),
                                        scale * gt_pos2) #第二帧误差
                #EMD
                emd1 = wasserstein_distance(pr_pos1.cpu().detach().numpy().flatten(), gt_pos1.flatten())
                emd2 = wasserstein_distance(pr_pos2.cpu().detach().numpy().flatten(), gt_pos2.flatten())
                emd_distance1.append(emd1)
                emd_distance2.append(emd2)

                # #momentum
                # delta_t = 0.02
                # # delta_x = torch.tensor(gt_pos1) - pr_pos1.cpu().detach()
                # # # v0 =  torch.from_numpy(frames[0]['vel0'][0])
                # # # acc = 2 * (delta_x - v0 * delta_t) / (delta_t ** 2)
                # # acc = delta_x / (delta_t**2)
                # # momentum_res  = torch.mean(acc, dim=0)
                # vel_res = torch.sum(pr_vel1, dim=0) - torch.sum(torch.tensor(frames[0]['vel0'][0]).to(device), dim=0)
                # num_points, _ = pr_vel1.size()
                # g = torch.tensor([0, -9.818, 0]).repeat(num_points, 1).to(device)
                # vel_res -= torch.sum(g, dim=0) * 0.02
                # print(vel_res)

                # #uncompressible
                # #不可压缩
                # rou1_pred = rou_calculate(pr_pos1.cpu().detach())
                # rou0 = rou_calculate(torch.tensor(frames[0]['pos0'][0]).to(device))
                # rou_res = torch.tensor(rou0) - torch.tensor(rou1_pred)
                # rou_res_mean = torch.mean(torch.abs(rou_res))
                # rou_array.append(rou_res_mean)

                # #max_density
                # dens1 = density_loss(torch.tensor(gt_pos1), pr_pos1.cpu().detach())
                # dens2 = density_loss(torch.tensor(gt_pos2), pr_pos2.cpu().detach())
            frames = []

    result = {}
    result['err_n1'] = np.mean(
        [v['mean'] for k, v in fluid_errors.errors.items() if k[1] + 1 == k[2]]) #第一帧误差
    result['err_n2'] = np.mean(
        [v['mean'] for k, v in fluid_errors.errors.items() if k[1] + 2 == k[2]]) #第二帧误差
    result['emd_n1'] = np.mean(emd_distance1)
    result['emd_n2'] = np.mean(emd_distance2)
    result['uncompress'] = np.mean(rou_array)
    print(result)
    print('done')

    return result


def evaluate_whole_sequence_torch(model,
                                  val_dataset,
                                  frame_skip,
                                  device,
                                  fluid_errors=None,
                                  scale=1):
    import torch
    print('evaluating.. ', end='')

    if fluid_errors is None:
        fluid_errors = FluidErrors()

    skip = frame_skip

    last_scene_id = None
    for data in val_dataset:
        scene_id = data['scene_id0'][0]
        if last_scene_id is None or last_scene_id != scene_id:
            print(scene_id, end=' ', flush=True)
            last_scene_id = scene_id
            box = torch.from_numpy(data['box'][0]).to(device)
            box_normals = torch.from_numpy(data['box_normals'][0]).to(device)
            init_pos = torch.from_numpy(data['pos0'][0]).to(device)
            init_vel = torch.from_numpy(data['vel0'][0]).to(device)

            inputs = (init_pos, init_vel, None, box, box_normals)
        else:
            inputs = (pr_pos, pr_vel, None, box, box_normals)

        pr_pos, pr_vel = model(inputs)

        frame_id = data['frame_id0'][0]
        if frame_id > 0 and frame_id % skip == 0:
            gt_pos = data['pos0'][0]
            fluid_errors.add_errors(scene_id,
                                    0,
                                    frame_id,
                                    scale * pr_pos.detach().cpu().numpy(),
                                    scale * gt_pos,
                                    compute_gt2pred_distance=True)

    result = {}
    result['whole_seq_err'] = np.mean([
        v['gt2pred_mean']
        for k, v in fluid_errors.errors.items()
        if 'gt2pred_mean' in v
    ])

    print(result)
    print('done')

    return result


def eval_checkpoint(checkpoint_path, val_files, fluid_errors, options, cfg):
    # checkpoint_path存放训练后的模型， val_files是测试集数据
    val_dataset = read_data_val(files=val_files, window=1, cache_data=True)

    if checkpoint_path.endswith('.index'):
        import tensorflow as tf

        model = trainscript.create_model(**cfg.get('model', {}))
        checkpoint = tf.train.Checkpoint(step=tf.Variable(0), model=model)
        checkpoint.restore(
            os.path.splitext(checkpoint_path)[0]).expect_partial()

        evaluate_tf(model, val_dataset, options.frame_skip, fluid_errors,
                    **cfg.get('evaluation', {}))
        evaluate_whole_sequence_tf(model, val_dataset, options.frame_skip,
                                   fluid_errors, **cfg.get('evaluation', {}))
    elif checkpoint_path.endswith('.h5'):
        import tensorflow as tf

        model = trainscript.create_model(**cfg.get('model', {}))
        model.init()
        model.load_weights(checkpoint_path, by_name=True)
        evaluate_tf(model, val_dataset, options.frame_skip, fluid_errors,
                    **cfg.get('evaluation', {}))
        evaluate_whole_sequence_tf(model, val_dataset, options.frame_skip,
                                   fluid_errors, **cfg.get('evaluation', {}))
    elif checkpoint_path.endswith('.pt'):
        import torch

        model = trainscript.create_model(**cfg.get('model', {}))
        checkpoint = torch.load(checkpoint_path)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model']) #读取训练好的模型
        else:
            model.load_state_dict(checkpoint)
        model.to(options.device)
        model.requires_grad_(False)
        evaluate_torch(model, val_dataset, options.frame_skip, options.device,
                       fluid_errors, **cfg.get('evaluation', {}))
        evaluate_whole_sequence_torch(model, val_dataset, options.frame_skip,
                                      options.device, fluid_errors,
                                      **cfg.get('evaluation', {}))
    else:
        raise Exception('Unknown checkpoint format')


def print_errors(fluid_errors):
    result = {}
    result['err_n1'] = np.mean(
        [v['mean'] for k, v in fluid_errors.errors.items() if k[1] + 1 == k[2]])
    result['err_n2'] = np.mean(
        [v['mean'] for k, v in fluid_errors.errors.items() if k[1] + 2 == k[2]])
    result['whole_seq_err'] = np.mean([
        v['gt2pred_mean']
        for k, v in fluid_errors.errors.items()
        if 'gt2pred_mean' in v
    ])
    print('====================\n', result)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluates a fluid network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--trainscript",
                        type=str,
                        required=True,
                        help="The python training script.")
    parser.add_argument("--cfg",
                        type=str,
                        required=True,
                        help="The path to the yaml config file")
    parser.add_argument(
        "--checkpoint_iter",
        type=int,
        required=False,
        help="The checkpoint iteration. The default is the last checkpoint.")
    parser.add_argument(
        "--weights",
        type=str,
        required=False,
        help="If set uses the specified weights file instead of a checkpoint.")
    parser.add_argument("--frame-skip",
                        type=int,
                        default=5,
                        help="The frame skip. Default is 5.")
    parser.add_argument("--device",
                        type=str,
                        default="cuda",
                        help="The device to use. Applies only for torch.")

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    global trainscript
    module_name = os.path.splitext(os.path.basename(args.trainscript))[0]
    sys.path.append('.')
    trainscript = importlib.import_module(module_name)

    train_dir = module_name + '_' + os.path.splitext(os.path.basename(
        args.cfg))[0]
    # val_files = sorted(glob(os.path.join(cfg['dataset_dir'], 'valid', '*.zst')))
    val_files = sorted(glob(os.path.join('../DeepLagrangianFluids-master/datasets/ours_6k_box_data', 'valid', '*.zst')))
    if args.weights is not None:
        print('evaluating :', args.weights)
        output_path = args.weights + '_6k_eval.json'
        if os.path.isfile(output_path):
            print('Printing previously computed results for :', args.weights,
                  output_path)
            fluid_errors = FluidErrors()
            fluid_errors.load(output_path)
        else:
            fluid_errors = FluidErrors()
            eval_checkpoint(args.weights, val_files, fluid_errors, args, cfg)
            fluid_errors.save(output_path)
    else:
        # get a list of checkpoints

        # tensorflow checkpoints
        checkpoint_files = glob(
            os.path.join(train_dir, 'checkpoints', 'ckpt-*.index'))
        # torch checkpoints
        checkpoint_files.extend(
            glob(os.path.join(train_dir, 'checkpoints', 'ckpt-*.pt')))
        all_checkpoints = sorted([
            (int(re.match('.*ckpt-(\d+)\.(pt|index)', x).group(1)), x)
            for x in checkpoint_files
        ]) #按照checkpoints文件名中的迭代次数排序

        # select the checkpoint
        if args.checkpoint_iter is not None:
            checkpoint = dict(all_checkpoints)[args.checkpoint_iter]
        else:
            checkpoint = all_checkpoints[-1]

        output_path = train_dir + '_6k_eval_{}.json'.format(checkpoint[0])
        if os.path.isfile(output_path):
            print('Printing previously computed results for :', checkpoint,
                  output_path) #如果上次evaluate的文件夹也叫这个名（模型经过的迭代数相同），则直接把之前的内容打印出来
            fluid_errors = FluidErrors()
            fluid_errors.load(output_path)
        else:
            print('evaluating :', checkpoint)
            fluid_errors = FluidErrors()
            eval_checkpoint(checkpoint[1], val_files, fluid_errors, args, cfg) #第一次evaluate
            fluid_errors.save(output_path)

    print_errors(fluid_errors)
    return 0


if __name__ == '__main__':
    sys.exit(main())
