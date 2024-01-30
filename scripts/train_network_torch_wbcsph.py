#!/usr/bin/env python3
import importlib
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import sys
import argparse
import yaml
from torch.backends import cudnn
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datasets.dataset_reader_physics import read_data_train, read_data_val
from collections import namedtuple
import glob
import time
import torch
from utils.deeplearningutilities.torch import Trainer, MyCheckpointManager
from evaluate_network import evaluate_torch

_k = 1000
device_ids = [0, 1]
TrainParams = namedtuple('TrainParams', ['max_iter', 'base_lr', 'batch_size'])
train_params = TrainParams(100 * _k, 0.001, 16)
# min_err = 0.00059
min_err = 0.00090

def create_model(**kwargs):
    from models.default_torch import MyParticleNetwork
    """Returns an instance of the network for training and evaluation"""
    model = MyParticleNetwork(**kwargs)
    return model


def main():
    global min_err
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(  "cfg",
                        type=str,
                        help="The path to the yaml config file")
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    # the train dir stores all checkpoints and summaries. The dir name is the name of this file combined with the name of the config file
    train_dir = os.path.splitext(
        os.path.basename(__file__))[0] + '_' + os.path.splitext(
            os.path.basename(args.cfg))[0]

    val_files = sorted(glob.glob(os.path.join(cfg['dataset_dir'], 'valid', '*.zst'))) #(20个场景序列，每个序列15帧，300帧)
    train_files = sorted(
        glob.glob(os.path.join(cfg['dataset_dir'], 'train', '*.zst'))) #(200个场景序列，每个序列15帧，共3000帧)

    device = torch.device("cuda")

    val_dataset = read_data_val(files=val_files, window=1, cache_data=True)

    dataset = read_data_train(files=train_files,
                              batch_size=train_params.batch_size,
                              window=3,
                              num_workers=2,
                              **cfg.get('train_data', {}))

    data_iter = iter(dataset) # 用iter迭代器来遍历dataset，iter通过next来遍历每一个batch

    trainer = Trainer(train_dir)

    model = create_model(**cfg.get('model', {}))
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = torch.nn.DataParallel(model, device_ids=[0,1])

    model.cuda()

    boundaries = [
        20 * _k,
        30 * _k,
        40 * _k,
        50 * _k,
        70 * _k,
        90 * _k,
        110 * _k
    ]
    # lr_values = [
    #     1.5,
    #     1.0,
    #     0.5,
    #     0.5,
    #     0.5,
    #     0.25
    # ]
    lr_values = [
        2.0,
        1.5,
        1.0,
        0.75,
        0.75,
        0.5,
        0.25
    ]

    def lrfactor_fn(x):
        factor = lr_values[0]
        for b, v in zip(boundaries, lr_values[1:]):
            if x > b:
                factor = v
            else:
                break
        return factor

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=train_params.base_lr,
                                 eps=1e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lrfactor_fn)

    step = torch.tensor(0)
    checkpoint_fn = lambda: {
        'step': step,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }

    manager = MyCheckpointManager(checkpoint_fn,
                                  trainer.checkpoint_dir,
                                  keep_checkpoint_steps=list(
                                      range(1 * _k, train_params.max_iter + 1,
                                            1 * _k)))

    def euclidean_distance(a, b, epsilon=1e-9):
        return torch.sqrt(torch.sum((a - b)**2, dim=-1) + epsilon)

    def loss_fn(pr_pos, gt_pos, num_fluid_neighbors):
        gamma = 0.5
        neighbor_scale = 1 / 40
        importance = torch.exp(-neighbor_scale * num_fluid_neighbors)
        return torch.mean(importance *
                          euclidean_distance(pr_pos, gt_pos)**gamma)

    def train(model, batch):
        optimizer.zero_grad()
        losses = []

        batch_size = train_params.batch_size
        for batch_i in range(batch_size):
            batch['pos0'][batch_i] = batch['pos0'][batch_i].to(torch.float32)
            batch['vel0'][batch_i] = batch['vel0'][batch_i].to(torch.float32)
            batch['pos1'][batch_i] = batch['pos1'][batch_i].to(torch.float32)
            batch['pos2'][batch_i] = batch['pos2'][batch_i].to(torch.float32)
            batch['box'][batch_i] = batch['box'][batch_i].to(torch.float32)
            batch['box_normals'][batch_i] = batch['box_normals'][batch_i].to(torch.float32)

            inputs = ([
                batch['pos0'][batch_i], batch['vel0'][batch_i], None,
                batch['box'][batch_i], batch['box_normals'][batch_i]
            ])
            # 一次传入一组场景数据的一帧 ———— 为什么不直接用一个batch计算？？？？？？？？？？？可能是因为该球形滤波器的特殊性，一次只能一帧的粒子卷积

            #第一帧
            pr_pos1, pr_vel1 = model(inputs)

            l = 0.5 * loss_fn(pr_pos1, batch['pos1'][batch_i],
                              model.num_fluid_neighbors)

            inputs = (pr_pos1, pr_vel1, None, batch['box'][batch_i],
                      batch['box_normals'][batch_i])
            #第二帧
            pr_pos2, pr_vel2 = model(inputs)

            l += 0.5 * loss_fn(pr_pos2, batch['pos2'][batch_i],
                               model.num_fluid_neighbors) # 两帧损失
            losses.append(l)

        total_loss = 128 * sum(losses) / batch_size
        total_loss.backward()
        optimizer.step() #一个batch更新一次

        return total_loss

    if manager.latest_checkpoint:
        print('restoring from ', manager.latest_checkpoint)
        latest_checkpoint = torch.load(manager.latest_checkpoint)
        step = latest_checkpoint['step']
        model.load_state_dict(latest_checkpoint['model'])
        optimizer.load_state_dict(latest_checkpoint['optimizer'])
        scheduler.load_state_dict(latest_checkpoint['scheduler'])

    display_str_list = []
    while trainer.keep_training(step,
                                train_params.max_iter, #最大迭代次数50000
                                checkpoint_manager=manager,
                                display_str_list=display_str_list):
        # 每次循环一个batch
        # 所以总共迭代帧应该是max_iter(50000)*batch_size
        # 通过不断迭代数据集直到停止，没有分多少个epoch，而是通过iter一直循环下去。 这里的iter()数据遍历完会继续重复
        data_fetch_start = time.time()
        batch = next(data_iter) # iter通过next来遍历每一个batch，其中一个batch有16(batch_size)组场景数据，每组数据包含20个属性，如pos1，vel1(第一帧gt)，pos2，vel2(第二帧gt)等 【每组数据只有两帧】

        batch_torch = {}
        for k in ('pos0', 'vel0', 'pos1', 'pos2', 'box', 'box_normals'):
            # batch_torch每组数据包含6个属性：pos0, vel0, pos1, pos2, box, box_normals
            batch_torch[k] = [torch.from_numpy(x).to(device) for x in batch[k]]
        data_fetch_latency = time.time() - data_fetch_start #数据延迟：存储或检索数据包所需的时间
        trainer.log_scalar_every_n_minutes(5, 'DataLatency', data_fetch_latency)

        current_loss = train(model, batch_torch)
        #batch_torch(6,64,4403,3) 6是6种属性（作键），64是batch_size，4403是该组场景数据中的点数（各场景点数不相同），3是特征维度（xyz）
        scheduler.step()
        display_str_list = ['loss', float(current_loss)] #记录每一次迭代的loss

        if trainer.current_step % 10 == 0:
            trainer.summary_writer.add_scalar('TotalLoss', current_loss,
                                              trainer.current_step)
            trainer.summary_writer.add_scalar('LearningRate',
                                              scheduler.get_last_lr()[0],
                                              trainer.current_step)
        # 每10个iteration打印一次loss

        if (trainer.current_step) % (1.0 * _k) == 0:
            for k, v in evaluate_torch(model,
                                 val_dataset,
                                 frame_skip=20,
                                 device=device,
                                 **cfg.get('evaluation', {})).items():
                trainer.summary_writer.add_scalar('eval/' + k, v,
                                                  trainer.current_step)
                if(k == "err_n1" and v < min_err):
                    min_err = v
                    torch.save({'model': model.state_dict()}, str(step.item())+'_model_weights_best.pt')
                    print("=================update best model: err_n1=" + str(v) + "===============")
        # 第1000次迭代eval一次

    torch.save({'model': model.state_dict()}, 'model_weights.pt') #所有batch遍历一遍记录下当前model，存到@/model_weights.pt
    if trainer.current_step == train_params.max_iter:
        return trainer.STATUS_TRAINING_FINISHED
    else:
        return trainer.STATUS_TRAINING_UNFINISHED


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn')
    sys.exit(main())
