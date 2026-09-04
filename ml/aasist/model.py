"""
AASIST Model Architecture
-------------------------
Adapted from the official clovaai/aasist repository:
  https://github.com/clovaai/aasist

Copyright (c) 2021-present NAVER Corp.
MIT license

Paper: "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal
        Graph Attention Networks" (ICASSP 2022)
  https://arxiv.org/abs/2110.01200

No modifications have been made to the model architecture.
"""

import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.0
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x):
        """x: (#bs, #node, #dim)"""
        x = self.input_drop(x)
        att_map = self._derive_att_map(x)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)
        return x * x_mirror

    def _derive_att_map(self, x):
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_map = torch.matmul(att_map, self.att_weight)
        att_map = att_map / self.temp
        att_map = F.softmax(att_map, dim=-2)
        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)
        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = 1.0
        if "temperature" in kwargs:
            self.temp = kwargs["temperature"]

    def forward(self, x1, x2, master=None):
        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)

        x1 = self.input_drop(x1)
        x2 = self.input_drop(x2)

        if master is None:
            master = torch.mean(torch.stack([x1, x2], dim=1), dim=1, keepdim=True)

        att_map = self._derive_att_map(x1, x2, master)
        x1 = self._project(x1, att_map[0])
        x2 = self._project(x2, att_map[1])
        master = self._project_master(master, att_map[2])

        x1 = self._apply_BN(x1)
        x2 = self._apply_BN(x2)
        master = self._apply_BN(master)

        x1 = self.act(x1)
        x2 = self.act(x2)
        master = self.act(master)

        return x1, x2, master

    def _derive_att_map(self, x1, x2, master):
        att_map11 = self._pairwise_mul_nodes(x1)
        att_map22 = self._pairwise_mul_nodes(x2)
        att_map12 = self._pairwise_mul_nodes_2types(x1, x2)

        att_map11 = torch.tanh(self.att_proj(att_map11))
        att_map22 = torch.tanh(self.att_proj(att_map22))
        att_map12 = torch.tanh(self.att_proj(att_map12))

        att_map11 = torch.matmul(att_map11, self.att_weight11)
        att_map22 = torch.matmul(att_map22, self.att_weight22)
        att_map12 = torch.matmul(att_map12, self.att_weight12)

        att_map11 = att_map11 / self.temp
        att_map22 = att_map22 / self.temp
        att_map12 = att_map12 / self.temp

        att_map11 = F.softmax(att_map11, dim=-2)
        att_map22 = F.softmax(att_map22, dim=-2)
        att_map12 = F.softmax(att_map12, dim=-2)

        # master node attention
        att_mapM1 = self._pairwise_mul_nodes_2types(master, x1)
        att_mapM2 = self._pairwise_mul_nodes_2types(master, x2)

        att_mapM1 = torch.tanh(self.att_projM(att_mapM1))
        att_mapM2 = torch.tanh(self.att_projM(att_mapM2))

        att_mapM1 = torch.matmul(att_mapM1, self.att_weightM)
        att_mapM2 = torch.matmul(att_mapM2, self.att_weightM)

        att_mapM1 = att_mapM1 / self.temp
        att_mapM2 = att_mapM2 / self.temp

        att_mapM1 = F.softmax(att_mapM1, dim=-2)
        att_mapM2 = F.softmax(att_mapM2, dim=-2)

        # Combined maps for x1, x2, master
        n1 = x1.size(1)
        n2 = x2.size(1)
        att_map_1 = torch.cat([att_map11, att_map12[:, :n1, :n2, :]], dim=2)
        att_map_2 = torch.cat([att_map12[:, :n2, :n1, :], att_map22], dim=2)
        att_map_M = torch.cat([att_mapM1, att_mapM2], dim=2)

        return att_map_1, att_map_2, att_map_M

    def _pairwise_mul_nodes(self, x):
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)
        return x * x_mirror

    def _pairwise_mul_nodes_2types(self, xa, xb):
        nb_nodes_a = xa.size(1)
        nb_nodes_b = xb.size(1)
        xa = xa.unsqueeze(2).expand(-1, -1, nb_nodes_b, -1)
        xb = xb.unsqueeze(1).expand(-1, nb_nodes_a, -1, -1)
        return xa * xb

    def _project(self, x, att_map):
        n = x.size(1)
        combined = torch.cat([x, x], dim=1)
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), combined))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _project_master(self, master, att_map):
        x_combined = torch.cat(
            [torch.mean(master, dim=1, keepdim=True)] * att_map.size(2), dim=2
        )
        x1 = self.proj_with_attM(
            torch.matmul(att_map.squeeze(-1), x_combined.transpose(1, 2))
        )
        x2 = self.proj_without_attM(master)
        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)
        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self.top_k_graph(scores, h, self.k)
        return new_h

    def top_k_graph(self, scores, h, k):
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)
        h = h * scores
        h = torch.gather(h, 1, idx)
        return h


class CONV(nn.Module):
    @staticmethod
    def to_mel(n_fft, f_min, f_max, sr, n_mels):
        import math

        def hz_to_mel(f):
            return 2595.0 * math.log10(1.0 + f / 700.0)

        def mel_to_hz(m):
            return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

        low_freq_mel = hz_to_mel(f_min)
        high_freq_mel = hz_to_mel(f_max)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
        hz_points = np.array([mel_to_hz(m) for m in mel_points])
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
        return bin_points

    def __init__(
        self,
        out_channels,
        in_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        sample_rate=16000,
        min_low_hz=50,
        min_band_hz=50,
    ):
        super().__init__()
        if in_channels != 1:
            msg = (
                f"SincConv only supports one input channel (here, in_channels = {in_channels})"
            )
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError("SincConv does not support bias.")
        if groups > 1:
            raise ValueError("SincConv does not support groups.")

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel_points = self.to_mel(
            self.kernel_size, low_hz, high_hz, self.sample_rate, self.out_channels
        )

        self.low_hz_ = nn.Parameter(
            torch.Tensor(mel_points[: self.out_channels]).view(-1, 1)
        )
        self.band_hz_ = nn.Parameter(
            torch.Tensor(
                np.diff(mel_points)[: self.out_channels]
            ).view(-1, 1)
        )

        n_lin = torch.linspace(
            0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2))
        )
        self.register_buffer("n_", n_lin.view(1, -1))

        window_ = 0.54 - 0.46 * torch.cos(
            2 * np.pi * n_lin / self.kernel_size
        )
        self.register_buffer("window_", window_.view(1, -1))

    def forward(self, waveforms):
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate / 2,
        )
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = (
            (torch.sin(f_times_t_high * 2 * np.pi) - torch.sin(f_times_t_low * 2 * np.pi))
            / (self.n_ / 2)
        ) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat(
            [band_pass_left, band_pass_center, band_pass_right], dim=1
        )
        band_pass = band_pass / (2.0 * band[:, None])

        self.filters = (band_pass).view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(
            waveforms,
            self.filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(
            in_channels=nb_filts[0],
            out_channels=nb_filts[1],
            kernel_size=(2, 3),
            padding=(1, 1),
            stride=1,
        )
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(
            in_channels=nb_filts[1],
            out_channels=nb_filts[1],
            kernel_size=(2, 3),
            padding=(0, 1),
            stride=1,
        )

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(
                in_channels=nb_filts[0],
                out_channels=nb_filts[1],
                padding=(0, 1),
                kernel_size=(1, 3),
                stride=1,
            )
        else:
            self.downsample = False

        self.mp = nn.MaxPool2d((1, 3))

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x
        out = self.conv1(x)
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)
        out += identity
        out = self.mp(out)
        return out


class Model(nn.Module):
    """
    AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal
    Graph Attention Networks

    Official architecture from clovaai/aasist (MIT License).
    """

    def __init__(self, d_args: dict):
        super().__init__()

        # Sinc-based convolution
        self.conv_time = CONV(
            out_channels=d_args["filts"][0],
            in_channels=1,
            kernel_size=d_args["first_conv"],
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False,
            padding_mode="zeros",
        )
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        # Residual blocks
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=d_args["filts"][1], first=True)),
            nn.Sequential(Residual_block(nb_filts=d_args["filts"][2])),
            nn.Sequential(Residual_block(nb_filts=d_args["filts"][3])),
            nn.Sequential(Residual_block(nb_filts=d_args["filts"][4])),
            nn.Sequential(Residual_block(nb_filts=d_args["filts"][4])),
            nn.Sequential(Residual_block(nb_filts=d_args["filts"][4])),
        )

        # Graph attention
        self.pos_S = nn.Parameter(torch.randn(1, 23, d_args["filts"][-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, d_args["filts"][-1][-1]))
        self.master2 = nn.Parameter(torch.randn(1, 1, d_args["filts"][-1][-1]))

        self.GAT_layer_S = GraphAttentionLayer(
            d_args["filts"][-1][-1], d_args["gat_dims"][0]
        )
        self.GAT_layer_T = GraphAttentionLayer(
            d_args["filts"][-1][-1], d_args["gat_dims"][0]
        )

        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            d_args["gat_dims"][0], d_args["gat_dims"][1], temperature=d_args["temperature"]
        )
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            d_args["gat_dims"][1], d_args["gat_dims"][1], temperature=d_args["temperature"]
        )

        self.pool_S = GraphPool(d_args["pool_ratios"][0], d_args["gat_dims"][0], 0.3)
        self.pool_T = GraphPool(d_args["pool_ratios"][1], d_args["gat_dims"][0], 0.3)
        self.pool_hS1 = GraphPool(d_args["pool_ratios"][2], d_args["gat_dims"][1], 0.3)
        self.pool_hT1 = GraphPool(d_args["pool_ratios"][2], d_args["gat_dims"][1], 0.3)

        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            d_args["gat_dims"][1], d_args["gat_dims"][1], temperature=d_args["temperature"]
        )
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            d_args["gat_dims"][1], d_args["gat_dims"][1], temperature=d_args["temperature"]
        )

        self.pool_hS2 = GraphPool(d_args["pool_ratios"][2], d_args["gat_dims"][1], 0.3)
        self.pool_hT2 = GraphPool(d_args["pool_ratios"][2], d_args["gat_dims"][1], 0.3)

        self.out_layer = nn.Linear(5 * d_args["gat_dims"][1], d_args["nb_classes"])

    def forward(self, x, Freq_aug=False):
        # Raw waveform encoder
        x = self.conv_time(x, )
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        # Encoder
        x = self.encoder(x)

        # Spectral / Temporal separation
        w = x.shape[-1]
        x_S = torch.mean(torch.abs(x[:, :, 2:, :]), dim=-1).transpose(1, 2)
        x_T = torch.mean(torch.abs(x[:, :, :2, :]), dim=-1).transpose(1, 2)

        # Positional encoding
        x_S = x_S + self.pos_S

        # GAT
        x_S = self.GAT_layer_S(x_S)
        x_T = self.GAT_layer_T(x_T)

        # Heterogeneous Graph Attention (first round)
        master1 = self.master1.expand(x_S.size(0), -1, -1)
        master2 = self.master2.expand(x_T.size(0), -1, -1)

        x_S, x_T, master1 = self.HtrgGAT_layer_ST11(x_S, x_T, master=master1)
        x_S1 = self.pool_S(x_S)
        x_T1 = self.pool_T(x_T)
        x_S, x_T, master1 = self.HtrgGAT_layer_ST12(x_S, x_T, master=master1)
        x_S1 = self.pool_hS1(x_S1)
        x_T1 = self.pool_hT1(x_T1)

        # Heterogeneous Graph Attention (second round)
        x_S, x_T, master2 = self.HtrgGAT_layer_ST21(x_S1, x_T1, master=master2)
        x_S2 = self.pool_hS2(x_S)
        x_T2 = self.pool_hT2(x_T)
        x_S, x_T, master2 = self.HtrgGAT_layer_ST22(x_S2, x_T2, master=master2)

        # Pooling & Output
        out_S = torch.max(torch.abs(x_S), dim=1)[0]
        out_T = torch.max(torch.abs(x_T), dim=1)[0]
        master1 = master1.squeeze(1)
        master2 = master2.squeeze(1)
        master1_T = torch.max(torch.abs(x_T1), dim=1)[0]

        out = self.out_layer(
            torch.cat([out_S, out_T, master1, master2, master1_T], dim=1)
        )
        return out
