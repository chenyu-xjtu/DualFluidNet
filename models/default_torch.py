import torch
import torch.nn.functional as F
import open3d.ml.torch as ml3d
import numpy as np
import torch.nn as nn
import copy
import math
from models.convolutions import ContinuousConv

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class AFF(nn.Module):
    def __init__(self, channels=32, inter_channels=64, conv_type='cconv'):
        super(AFF, self).__init__()
        self.filter_extent = torch.tensor(np.float32(1.5 * 6 * 0.025))
        def window_poly6(r_sqr):
            return torch.clamp((1 - r_sqr) ** 3, 0, 1)  # torch.clamp()将输入input张量每个元素的范围限制到区间 [min,max]，返回结果到一个新张量。

        def Conv(name, activation=None, conv_type='cconv', **kwargs):
            if conv_type == 'cconv':
                conv_fn = ml3d.layers.ContinuousConv
            elif conv_type == 'ascc':
                conv_fn = ContinuousConv
            conv = conv_fn(kernel_size=[4, 4, 4],
                           activation=activation,
                           align_corners=True,
                           normalize=False,
                           window_function=window_poly6,
                           radius_search_ignore_query_points=True,
                           **kwargs)
            return conv

        #一次aff
        self.cconv1 = Conv(name="conv1",
             in_channels=channels*2,
             filters=inter_channels,
             activation=None,
             conv_type=conv_type)
        self.batchNorm1 = nn.BatchNorm1d(inter_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.cconv2 = Conv(name="conv2",
                           in_channels=inter_channels,
                           filters=channels,
                           activation=None,
                           conv_type=conv_type)
        self.batchNorm2 = nn.BatchNorm1d(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, pos):
        xa = torch.cat((x, y), -1)
        # xa = torch.unsqueeze(xa, dim=0).transpose(1, 2)
        # xa = torch.unsqueeze(xa, dim=0)

        #第一次aff
        xl = self.cconv1(xa, pos, pos, self.filter_extent)
        xl = self.batchNorm1(xl)
        xl = self.relu1(xl)
        xl = self.cconv2(xl, pos, pos, self.filter_extent)
        xl = self.batchNorm2(xl)
        wei1 = self.sigmoid(xl)
        xo = 2 * x * wei1 + 2 * y * (1 - wei1)
        return xo

class IAFF(nn.Module):
    def __init__(self, channels=32, inter_channels=64, conv_type='cconv'):
        super(IAFF, self).__init__()
        self.filter_extent = torch.tensor(np.float32(1.5 * 6 * 0.025))
        def window_poly6(r_sqr):
            return torch.clamp((1 - r_sqr) ** 3, 0, 1)  # torch.clamp()将输入input张量每个元素的范围限制到区间 [min,max]，返回结果到一个新张量。

        def Conv(name, activation=None, conv_type='cconv', **kwargs):
            if conv_type == 'cconv':
                conv_fn = ml3d.layers.ContinuousConv
            elif conv_type == 'ascc':
                conv_fn = ContinuousConv
            conv = conv_fn(kernel_size=[4, 4, 4],
                           activation=activation,
                           align_corners=True,
                           normalize=False,
                           window_function=window_poly6,
                           radius_search_ignore_query_points=True,
                           **kwargs)
            return conv

        #第一次aff
        self.cconv1 = Conv(name="conv1",
             in_channels=channels*2,
             filters=inter_channels,
             activation=None,
             conv_type=conv_type)
        self.batchNorm1 = nn.BatchNorm1d(inter_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.cconv2 = Conv(name="conv2",
                           in_channels=inter_channels,
                           filters=channels,
                           activation=None,
                           conv_type=conv_type)
        self.batchNorm2 = nn.BatchNorm1d(channels)

        #第二次aff
        self.cconv3 = Conv(name="conv3",
             in_channels=channels,
             filters=inter_channels,
             activation=None,
             conv_type=conv_type)
        self.batchNorm3 = nn.BatchNorm1d(inter_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.cconv4 = Conv(name="conv4",
                           in_channels=inter_channels,
                           filters=channels,
                           activation=None,
                           conv_type=conv_type)
        self.batchNorm4 = nn.BatchNorm1d(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y, pos):
        xa = torch.cat((x, y), -1)
        # xa = torch.unsqueeze(xa, dim=0).transpose(1, 2)
        # xa = torch.unsqueeze(xa, dim=0)

        #第一次aff
        xl = self.cconv1(xa, pos, pos, self.filter_extent)
        xl = self.batchNorm1(xl)
        xl = self.relu1(xl)
        xl = self.cconv2(xl, pos, pos, self.filter_extent)
        xl = self.batchNorm2(xl)
        wei1 = self.sigmoid(xl)
        xo = 2 * x * wei1 + 2 * y * (1 - wei1)
        #第二次aff
        xo = self.cconv3(xo, pos, pos, self.filter_extent)
        xo = self.batchNorm3(xo)
        xo = self.relu2(xo)
        xo = self.cconv4(xo, pos, pos, self.filter_extent)
        xo = self.batchNorm4(xo)
        wei2 = self.sigmoid(xo)
        # wei = torch.squeeze(wei, dim=0)
        # wei = torch.squeeze(wei, dim=0).transpose(0, 1)
        xo = 2 * x * wei2 + 2 * y * (1 - wei2)
        return xo

class MyParticleNetwork(torch.nn.Module):

    def __init__(
            self,
            kernel_size=[4, 4, 4],
            radius_scale=1.5,
            coordinate_mapping='ball_to_cube_volume_preserving',
            interpolation='linear',
            use_window=True,
            particle_radius=0.025,
            timestep=1 / 50,
            gravity=(0, -9.81, 0),
            other_feats_channels=0,
    ):
        super().__init__()
        self.layer_channels = [32, 64, 128, 64, 3]
        self.kernel_size = kernel_size
        self.radius_scale = radius_scale
        self.coordinate_mapping = coordinate_mapping
        self.interpolation = interpolation
        self.use_window = use_window
        self.particle_radius = particle_radius
        self.filter_extent = np.float32(self.radius_scale * 6 *
                                        self.particle_radius)
        self.timestep = timestep
        gravity = torch.FloatTensor(gravity)
        self.register_buffer('gravity', gravity)
        # self.transformer = Transformer(emb_dims=32, n_blocks=1, dropout=0.0 ,ff_dims=64, n_heads=1, final_dims=64)


        def window_poly6(r_sqr):
            return torch.clamp((1 - r_sqr)**3, 0, 1) #torch.clamp()将输入input张量每个元素的范围限制到区间 [min,max]，返回结果到一个新张量。

        def Conv(name, activation=None, conv_type='cconv', **kwargs):
            if conv_type == 'cconv':
                conv_fn = ml3d.layers.ContinuousConv
            elif conv_type == 'ascc':
                conv_fn = ContinuousConv
            window_fn = None
            if self.use_window == True:
                window_fn = window_poly6

            conv = conv_fn(kernel_size=self.kernel_size,
                           activation=activation,
                           align_corners=True,
                           interpolation=self.interpolation,
                           coordinate_mapping=self.coordinate_mapping,
                           normalize=False,
                           window_function=window_fn,
                           radius_search_ignore_query_points=True,
                           **kwargs)
            if conv_type == 'cconv':
                self._all_convs_cconv.append((name, conv))
            elif conv_type == 'ascc':
                self._all_convs_ascc.append((name, conv))
            return conv

    #cconv
        self.aff_cconv = IAFF(channels=32, inter_channels=64, conv_type='cconv')
        self._all_convs_cconv = []
        #定义第一层的三个卷积
        self.conv0_fluid_cconv = Conv(name="cconv0_fluid",
                                in_channels=4 + other_feats_channels,
                                filters=self.layer_channels[0],
                                activation=None,
                                conv_type='cconv')
        self.conv0_obstacle_cconv = Conv(name="cconv0_obstacle",
                                   in_channels=3,
                                   filters=self.layer_channels[0],
                                   activation=None,
                                   conv_type='cconv')
        self.dense0_fluid_cconv = torch.nn.Linear(in_features=4 +
                                            other_feats_channels,
                                            out_features=self.layer_channels[0])
        torch.nn.init.xavier_uniform_(self.dense0_fluid_cconv.weight)
        torch.nn.init.zeros_(self.dense0_fluid_cconv.bias)

        self.convs_cconv = []
        self.denses_cconv = []
        for i in range(1, len(self.layer_channels)):
            #从第二层开始定义每层的卷积conv或全连接层dense，存到convs[]和denses[]中
            in_ch = self.layer_channels[i - 1]
            if i == 1:
                # in_ch *= 3 #第二层的输入维度为3个32
                in_ch = 64 #第二层的输入维度为128
            out_ch = self.layer_channels[i]
            dense_cconv = torch.nn.Linear(in_features=in_ch, out_features=out_ch)
            torch.nn.init.xavier_uniform_(dense_cconv.weight)
            torch.nn.init.zeros_(dense_cconv.bias)
            setattr(self, 'dense_cconv{0}'.format(i), dense_cconv)
            conv = Conv(name='cconv{0}'.format(i),
                        in_channels=in_ch,
                        filters=out_ch,
                        activation=None,
                        conv_type='cconv')
            setattr(self, 'cconv{0}'.format(i), conv)
            self.denses_cconv.append(dense_cconv)
            self.convs_cconv.append(conv)

    #ASCC
        self.aff_ascc = IAFF(channels=32, inter_channels=64, conv_type='ascc')
        self._all_convs_ascc = []
        # 定义第一层的三个卷积
        self.conv0_fluid_ascc = Conv(name="ascc0_fluid",
                                in_channels=4 + other_feats_channels,
                                filters=self.layer_channels[0],
                                activation=None,
                                conv_type='ascc')
        self.conv0_obstacle_ascc = Conv(name="ascc0_obstacle",
                                   in_channels=3,
                                   filters=self.layer_channels[0],
                                   activation=None,
                                   conv_type='ascc')
        self.dense0_fluid_ascc = torch.nn.Linear(in_features=4 +
                                                        other_feats_channels,
                                            out_features=self.layer_channels[0])
        torch.nn.init.xavier_uniform_(self.dense0_fluid_ascc.weight)
        torch.nn.init.zeros_(self.dense0_fluid_ascc.bias)

        self.convs_ascc = []
        self.denses_ascc = []
        for i in range(1, len(self.layer_channels)):
            # 从第二层开始定义每层的卷积conv或全连接层dense，存到convs[]和denses[]中
            in_ch = self.layer_channels[i - 1]
            if i == 1:
                # in_ch *= 3 #第二层的输入维度为3个32
                in_ch = 64  # 第二层的输入维度为128
            out_ch = self.layer_channels[i]
            dense_ascc = torch.nn.Linear(in_features=in_ch, out_features=out_ch)
            torch.nn.init.xavier_uniform_(dense_ascc.weight)
            torch.nn.init.zeros_(dense_ascc.bias)
            setattr(self, 'dense_ascc{0}'.format(i), dense_ascc)
            conv_ascc = Conv(name='ascc{0}'.format(i),
                        in_channels=in_ch,
                        filters=out_ch,
                        activation=None,
                        conv_type='ascc')
            setattr(self, 'ascc{0}'.format(i), conv_ascc)
            self.denses_ascc.append(dense_ascc)
            self.convs_ascc.append(conv_ascc)
    # AFF
        self.affs = []
        self.aff0 = AFF(channels=self.layer_channels[0]*2, inter_channels=self.layer_channels[0]*2, conv_type='cconv')
        for i in range(1, len(self.layer_channels)):
            ch = self.layer_channels[i]
            aff = AFF(channels=ch, inter_channels=ch, conv_type='cconv')
            setattr(self, 'aff'+str(i), aff)
            self.affs.append(aff)
        self.resAff = AFF(channels=64, inter_channels=64, conv_type='cconv')
    def integrate_pos_vel(self, pos1, vel1):
        """Apply gravity and integrate position and velocity"""
        dt = self.timestep
        vel2 = vel1 + dt * self.gravity
        pos2 = pos1 + dt * (vel2 + vel1) / 2
        return pos2, vel2

    def compute_new_pos_vel(self, pos1, vel1, pos2, vel2, pos_correction):
        """Apply the correction
        pos1,vel1 are the positions and velocities from the previous timestep
        pos2,vel2 are the positions after applying gravity
        """
        dt = self.timestep
        pos = pos2 + pos_correction
        # vel = 2 * (pos - pos1) / dt - vel1
        vel = (pos - pos1) / dt
        return pos, vel

    def compute_correction(self,
                           pos,
                           vel,
                           other_feats,
                           box,
                           box_feats,
                           fixed_radius_search_hash_table=None):
        """Expects that the pos and vel has already been updated with gravity and velocity"""

        # compute the extent of the filters (the diameter)
        filter_extent = torch.tensor(self.filter_extent) # self.filter_extent = np.float32(self.radius_scale * 6 * self.particle_radius) = 1.5 * 6 * 0.025
        fluid_feats = [torch.ones_like(pos[:, 0:1]), vel] # ones_like函数：根据给定张量，生成与其形状相同的全1张量
        # fluid_feats [(4404,1), (4404,3)] 其中(4404,1)为全1, (4404,3)为速度
        if not other_feats is None:
            fluid_feats.append(other_feats)
        fluid_feats = torch.cat(fluid_feats, axis=-1) #特征向量[1,v] (4404,4) #没有粘度viscosity
    #cconv
        # 经过第一层网络
        self.ans_conv0_fluid_cconv = self.conv0_fluid_cconv(fluid_feats, pos, pos,
                                                filter_extent) # (4404,32)
        self.ans_dense0_fluid_cconv = self.dense0_fluid_cconv(fluid_feats) #（4404,32）
        self.ans_conv0_obstacle_cconv = self.conv0_obstacle_cconv(box_feats, box, pos,
                                                      filter_extent) #（4404,32）
        self.ans_dense0_fluid_cconv = self.dense0_fluid_cconv(fluid_feats)

        #IAFF
        self.hybrid_aff_cconv = self.aff_cconv(self.ans_conv0_fluid_cconv, self.ans_conv0_obstacle_cconv, pos) #(4404, 32)

        feats_cconv = torch.cat([
            self.hybrid_aff_cconv, self.ans_dense0_fluid_cconv
        ], axis=-1)  # （4404,64）
    # ascc
        # 经过第一层网络
        self.ans_conv0_fluid_ascc = self.conv0_fluid_ascc(fluid_feats, pos, pos,
                                                          filter_extent)  # (4404,32)
        self.ans_dense0_fluid_ascc = self.dense0_fluid_ascc(fluid_feats)  # （4404,32）
        self.ans_conv0_obstacle_ascc = self.conv0_obstacle_ascc(box_feats, box, pos,
                                                                filter_extent)  # （4404,32）
        self.ans_dense0_fluid_ascc = self.dense0_fluid_ascc(fluid_feats)

        # IAFF
        self.hybrid_aff_ascc = self.aff_ascc(self.ans_conv0_fluid_ascc, self.ans_conv0_obstacle_ascc, pos)  # (4404, 32)

        feats_ascc = torch.cat([
            self.hybrid_aff_ascc, self.ans_dense0_fluid_ascc
        ], axis=-1)  # （4404,64）

        feats_select = self.aff0(feats_cconv, feats_ascc, pos)

        self.ans_convs = [feats_select]

        for conv_cconv, dense_cconv, conv_ascc, dense_ascc, aff in zip(self.convs_cconv, self.denses_cconv, self.convs_ascc, self.denses_ascc, self.affs):
            #经过后三层网络，每层的结果存在ans_convs[]中
            inp_feats = F.relu(self.ans_convs[-1])
            #cconv
            ans_conv_cconv = conv_cconv(inp_feats, pos, pos, filter_extent)
            ans_dense_cconv = dense_cconv(inp_feats)
            ans_cconv = ans_conv_cconv + ans_dense_cconv
            #ascc
            ans_conv_ascc = conv_ascc(inp_feats, pos, pos, filter_extent)
            ans_dense_ascc = dense_ascc(inp_feats)
            ans_ascc = ans_conv_ascc + ans_dense_ascc
            #aff
            ans_select = aff(ans_cconv, ans_ascc, pos)
            #ResAFF
            if len(self.ans_convs) == 3 and ans_dense_cconv.shape[-1] == self.ans_convs[-2].shape[-1]:
                ans_select = self.resAff(ans_select, self.ans_convs[-2], pos)
                # print("ResAff")
            self.ans_convs.append(ans_select)

        # compute the number of fluid neighbors.
        # this info is used in the loss function during training.
        self.num_fluid_neighbors = ml3d.ops.reduce_subarrays_sum(
            torch.ones_like(self.conv0_fluid_cconv.nns.neighbors_index,
                            dtype=torch.float32),
            self.conv0_fluid_cconv.nns.neighbors_row_splits)

        # scale to better match the scale of the output distribution
        self.pos_correction = (1.0 / 128) * self.ans_convs[-1]

        return self.pos_correction

    def forward(self, inputs, fixed_radius_search_hash_table=None):
        """computes 1 simulation timestep
        inputs: list or tuple with (pos,vel,feats,box,box_feats)
          pos and vel are the positions and velocities of the fluid particles.
          feats is reserved for passing additional features, use None here.
          box are the positions of the static particles and box_feats are the
          normals of the static particles.
        """
        # 因为训练过程中是将每个batch中的各个场景数据分开传入model，所以这里得到的inputs是一组场景数据，如（6,4403,3），其中6是6种属性（作键），4403是该组场景数据中的点数（各场景点数不相同），3是特征维度（xyz）
        pos, vel, feats, box, box_feats = inputs # box是边界，box_feats是边界粒子的法向量, feats在这里还是None
        # pos,vel属性都是（4403,3）, box，box_feats是（38706,3）
        pos2, vel2 = self.integrate_pos_vel(pos, vel) # 施加重力后的速度和位置
        pos_correction = self.compute_correction(
            pos2, vel2, feats, box, box_feats, fixed_radius_search_hash_table) # 经过网络得到deltax
        pos2_corrected, vel2_corrected = self.compute_new_pos_vel(
            pos, vel, pos2, vel2, pos_correction)  # 原来的x和v加上deltax更新最后的x和v

        return pos2_corrected, vel2_corrected

    # def init(self, feats_shape=None):
    # """Runs the network with dummy data to initialize the shape of all variables"""
    # pos = np.zeros(shape=(1, 3), dtype=np.float32)
    # vel = np.zeros(shape=(1, 3), dtype=np.float32)
    # if feats_shape is None:
    # feats = None
    # else:
    # feats = np.zeros(shape=feats_shape, dtype=np.float32)
    # box = np.zeros(shape=(1, 3), dtype=np.float32)
    # box_feats = np.zeros(shape=(1, 3), dtype=np.float32)

    # _ = self.__call__((pos, vel, feats, box, box_feats))
