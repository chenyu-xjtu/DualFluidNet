import torch
import torch.nn.functional as F
import open3d.ml.torch as ml3d
import numpy as np
import torch.nn as nn
import copy
import math

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


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


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=None):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))

class FinalLayerConnection(nn.Module):
    def __init__(self, size, dropout=None):
        super(FinalLayerConnection, self).__init__()
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer):
        return sublayer(self.norm(x))
class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class MyAttention(nn.Module):

    def __init__(self, emb_dims, final_dims, self_attn, src_attn, feed_forward, dropout):
        super(MyAttention, self).__init__()
        self.emb_dims = emb_dims
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(emb_dims, dropout), 2)
        # self.sublayer = SublayerConnection(emb_dims, dropout)
        self.finallayer = FinalLayerConnection(emb_dims, dropout)

    def forward(self, fluid, obstacle, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        hybrid_embedding = self.sublayer[0](fluid, lambda fluid: self.self_attn(fluid, fluid, fluid, None))
        hybrid_embedding = self.sublayer[1](fluid, lambda obstacle: self.src_attn(obstacle, hybrid_embedding, hybrid_embedding, None)) #(32)
        # hybrid_embedding = self.sublayer(fluid, lambda fluid: self.self_attn(fluid, obstacle, fluid, None))
        return self.finallayer(hybrid_embedding, self.feed_forward) #64
        # return hybrid_embedding
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = None

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        #cconv没有batch
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(-1, self.h, self.d_k).transpose(0, 1).contiguous()
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(0, 1).contiguous() \
            .view(-1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, final_dims, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.norm = nn.Sequential()  # nn.BatchNorm1d(d_ff)
        self.w_2 = nn.Linear(d_ff, final_dims)
        self.dropout = None

    def forward(self, x):
        return self.w_2(self.norm(F.relu(self.w_1(x)).transpose(0, 1).contiguous()).transpose(0, 1).contiguous())

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        # src_mask, tgt_mask = None
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        #memory：self.encoder(self.src_embed(src), src_mask)
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))

class Transformer(nn.Module):
    def __init__(self, emb_dims, n_blocks, dropout, ff_dims, n_heads, final_dims):
        super(Transformer, self).__init__()
        self.emb_dims = emb_dims
        self.N = n_blocks
        self.dropout = dropout
        self.ff_dims = ff_dims
        self.n_heads = n_heads
        self.final_dims = final_dims
        c = copy.deepcopy
        # copy.deepcopy()的用法是将某一个变量的值赋值给另一个变量(此时两个变量地址不同)，因为地址不同，所以可以防止变量间相互干扰。
        # copy.copy()是浅拷贝，只拷贝父对象，不会拷贝对象的内部的子对象。copy.deepcopy()是深拷贝，会拷贝对象及其子对象
        attn = MultiHeadedAttention(self.n_heads, self.emb_dims)
        ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.final_dims, self.dropout)
        # self.model_transformer = EncoderDecoder(Encoder(EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N),
        #                             Decoder(DecoderLayer(self.emb_dims, c(attn), c(attn), c(ff), self.dropout), self.N),
        #                             nn.Sequential(),
        #                             nn.Sequential(),
        #                             nn.Sequential())
        self.myAttention = MyAttention(self.emb_dims, self.final_dims, c(attn), c(attn), c(ff), self.dropout)
    def forward(self, *input):
        fluid = input[0] #FX
        obstacle = input[1] #FY
        # src = src.transpose(2, 1).contiguous()
        # tgt = tgt.transpose(2, 1).contiguous()
        fluid = fluid.contiguous()
        obstacle = obstacle.contiguous()
        # transpose(2,1)调换第一维和第二维
        # contiguous()函数会使tensor变量在内存中的存储变得连续。
        # view只能用在contiguous的variable上。如果在view之前用了transpose, permute等，需要用contiguous()来返回一个contiguous copy。
        # tgt_embedding = self.model_transformer(src, tgt, None, None).transpose(2, 1).contiguous()
        # src_embedding = self.model_transformer(tgt, src, None, None).transpose(2, 1).contiguous()

        # hybrid_embedding = self.model_transformer(fluid, obstacle, None, None).contiguous()
        hybrid_embedding = self.myAttention(fluid, obstacle, None, None).contiguous()
        return hybrid_embedding

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
        self.layer_channels = [32, 64, 64, 3]
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
        self.transformer = Transformer(emb_dims=32, n_blocks=1, dropout=0.0 ,ff_dims=64, n_heads=1, final_dims=64)
        self._all_convs = []

        def window_poly6(r_sqr):
            return torch.clamp((1 - r_sqr)**3, 0, 1) #torch.clamp()将输入input张量每个元素的范围限制到区间 [min,max]，返回结果到一个新张量。

        def Conv(name, activation=None, **kwargs):
            conv_fn = ml3d.layers.ContinuousConv #封装好的卷积函数

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
            self._all_convs.append((name, conv))
            return conv

        #定义第一层的三个卷积
        self.conv0_fluid = Conv(name="conv0_fluid",
                                in_channels=4 + other_feats_channels,
                                filters=self.layer_channels[0],
                                activation=None)
        self.conv0_obstacle = Conv(name="conv0_obstacle",
                                   in_channels=3,
                                   filters=self.layer_channels[0],
                                   activation=None)
        # self.dense0_fluid = torch.nn.Linear(in_features=4 +
        #                                     other_feats_channels,
        #                                     out_features=self.layer_channels[0])
        # torch.nn.init.xavier_uniform_(self.dense0_fluid.weight)
        # torch.nn.init.zeros_(self.dense0_fluid.bias)

        self.convs = []
        self.denses = []
        for i in range(1, len(self.layer_channels)):
            #从第二层开始定义每层的卷积conv或全连接层dense，存到convs[]和denses[]中
            in_ch = self.layer_channels[i - 1]
            if i == 1:
                # in_ch *= 3 #第二层的输入维度为3个32
                in_ch = 64 #第二层的输入维度为128
            out_ch = self.layer_channels[i]
            dense = torch.nn.Linear(in_features=in_ch, out_features=out_ch)
            torch.nn.init.xavier_uniform_(dense.weight)
            torch.nn.init.zeros_(dense.bias)
            setattr(self, 'dense{0}'.format(i), dense)
            conv = Conv(name='conv{0}'.format(i),
                        in_channels=in_ch,
                        filters=out_ch,
                        activation=None)
            setattr(self, 'conv{0}'.format(i), conv)
            self.denses.append(dense)
            self.convs.append(conv)

    def integrate_pos_vel(self, pos1, vel1):
        """Apply gravity and integrate position and velocity"""
        dt = self.timestep
        vel2 = vel1 + dt * self.gravity
        pos2 = pos1 + dt * (vel2 + vel1) / 2
        return pos2, vel2

    def compute_new_pos_vel(self, pos1, vel1, pos2, vel2, pos_correction):
        """Apply the correction
        pos1,vel1 are the positions and velocities from the previous timestep
        pos2,vel2 are the positions after applying gravity and the integration step
        """
        dt = self.timestep
        pos = pos2 + pos_correction
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

        # 经过第一层网络
        self.ans_conv0_fluid = self.conv0_fluid(fluid_feats, pos, pos,
                                                filter_extent) # (4404,32)
        # self.ans_dense0_fluid = self.dense0_fluid(fluid_feats) #（4404,32）
        self.ans_conv0_obstacle = self.conv0_obstacle(box_feats, box, pos,
                                                      filter_extent) #（4404,32）
        # self.ans_dense0_fluid = self.dense0_fluid(fluid_feats)
        #transformer
        self.hybrid_embedding = self.transformer(self.ans_conv0_fluid, self.ans_conv0_obstacle) #（4404,128)

        # (inp_features, inp_positions, out_positions, extents)
        # extents: The extent defines the spatial size of the filter for each output point.
        # 这里的inp_positions和out_positions怎么理解???????
        # feats = torch.cat([
        #     self.ans_conv0_obstacle, self.ans_conv0_fluid, self.ans_dense0_fluid
        # ], axis=-1) #（4404,96）

        feats = self.hybrid_embedding #(4404,128)
        # feats = torch.cat([
        #     self.hybrid_embedding, self.ans_dense0_fluid
        # ], axis=-1)  # （4404,64）

        self.ans_convs = [feats]
        for conv, dense in zip(self.convs, self.denses):
            #经过后三层网络，每层的结果存在ans_convs[]中
            inp_feats = F.relu(self.ans_convs[-1])
            ans_conv = conv(inp_feats, pos, pos, filter_extent)
            ans_dense = dense(inp_feats)
            if ans_dense.shape[-1] == self.ans_convs[-1].shape[-1]:
                ans = ans_conv + ans_dense + self.ans_convs[-1]
            else:
                ans = ans_conv + ans_dense
            self.ans_convs.append(ans)

        # compute the number of fluid neighbors.
        # this info is used in the loss function during training.
        self.num_fluid_neighbors = ml3d.ops.reduce_subarrays_sum(
            torch.ones_like(self.conv0_fluid.nns.neighbors_index,
                            dtype=torch.float32),
            self.conv0_fluid.nns.neighbors_row_splits)

        self.last_features = self.ans_convs[-2]

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
