"""Functions for reading the compressed training/validation data records"""
import os
import sys
import numpy as np
from glob import glob
import dataflow
import numpy as np
import zstandard as zstd
import msgpack
import msgpack_numpy
msgpack_numpy.patch()


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


def read_data_val(files, **kwargs):
    return read_data(files=files,
                     batch_size=1,
                     repeat=False,
                     shuffle_buffer=None,
                     num_workers=1,
                     **kwargs)


def read_data_train(files, batch_size, random_rotation=True, **kwargs):
    return read_data(files=files,
                     batch_size=batch_size,
                     random_rotation=random_rotation,
                     repeat=True,
                     shuffle_buffer=512,
                     **kwargs) #训练时随机旋转，测试时不旋转


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
