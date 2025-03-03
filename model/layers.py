import sys
sys.path.append("..")

import torch.nn as nn
from torch.nn import functional as F
import torch
from functools import partial
from torch.nn import Conv3d, Module, Linear, BatchNorm3d, ReLU
from torch.nn.modules.utils import _pair, _triple

import torch.nn.modules.conv as conv


class AddCoords(nn.Module):
    def __init__(self, rank, with_r=False):
        super(AddCoords, self).__init__()
        self.rank = rank
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """
        if self.rank == 1:
            batch_size_shape, channel_in_shape, dim_x = input_tensor.shape
            x_range = torch.linspace(-1, 1, dim_x, device=input_tensor.device)
            xx_channel = x_range.expand([batch_size_shape, 1, -1])

            out = torch.cat([input_tensor, xx_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 2:
            batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
            x_range = torch.linspace(-1, 1, dim_x, device=input_tensor.device)
            y_range = torch.linspace(-1, 1, dim_y, device=input_tensor.device)
            yy_channel, xx_channel = torch.meshgrid(y_range, x_range)
            yy_channel = yy_channel.expand([batch_size_shape, 1, -1, -1])
            xx_channel = xx_channel.expand([batch_size_shape, 1, -1, -1])

            out = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) + torch.pow(yy_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 3:
            batch_size_shape, channel_in_shape, dim_z, dim_y, dim_x = input_tensor.shape
            x_range = torch.linspace(-1, 1, dim_x, device=input_tensor.device)
            y_range = torch.linspace(-1, 1, dim_y, device=input_tensor.device)
            z_range = torch.linspace(-1, 1, dim_z, device=input_tensor.device)
            zz_channel, yy_channel, xx_channel = torch.meshgrid(z_range, y_range, x_range)
            zz_channel = zz_channel.expand([batch_size_shape, 1, -1, -1, -1])
            yy_channel = yy_channel.expand([batch_size_shape, 1, -1, -1, -1])
            xx_channel = xx_channel.expand([batch_size_shape, 1, -1, -1, -1])
            out = torch.cat([input_tensor, xx_channel, yy_channel, zz_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) +
                                torch.pow(yy_channel - 0.5, 2) +
                                torch.pow(zz_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)
        else:
            raise NotImplementedError

        return out


def normalization(planes, norm):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        #m = nn.InstanceNorm3d(planes)
        # my modification, this is the norm used in TomoTwin
        m = nn.GroupNorm(planes, planes, eps=1e-05, affine=True)
    # elif norm == 'sync_bn'
    #     m = SynchronizedBatchNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m


class LatentToFILM(nn.Module):
    def __init__(self, out_dim: int, in_dim: int=32, activation=nn.LeakyReLU()) -> None:
        super().__init__()
        if activation is None:
            activation = nn.Identity()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            #nn.GroupNorm(out_dim, out_dim, eps=1e-05, affine=True),
            activation,
        )
    
    def forward(self, x):
        return self.layers(x)


class ExtResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm='bn', act='relu', promptable=False, prompt_dim=32, film_activation=None):
        super(ExtResNetBlock, self).__init__()
        # first convolution
        self.conv1 = SingleConv(in_channels, out_channels, norm=norm, act=act)
        # residual block
        self.conv2 = SingleConv(out_channels, out_channels, norm=norm, act=act)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        self.conv3 = SingleConv(out_channels, out_channels, norm=norm, act=act)
        self.non_linearity = nn.ELU(inplace=False)
        self.promptable = promptable
        if self.promptable:
            self.film_scale_layer = LatentToFILM(in_dim=prompt_dim, out_dim=out_channels, activation=film_activation)
            self.film_loc_layer = LatentToFILM(in_dim=prompt_dim, out_dim=out_channels, activation=film_activation)

    def forward(self, x, prompt=None):
        # apply first convolution and save the output as a residual
        out = self.conv1(x)
        residual = out
        # residual block
        out = self.conv2(out)
        
        if self.promptable and prompt is not None:
            film_slope = self.film_scale_layer(prompt).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            film_intercept = self.film_loc_layer(prompt).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            out = out * film_slope + film_intercept
        
        out = self.conv3(out)

        out = out + residual
        out = self.non_linearity(out)
        return out

class ExtResNetBlock_lightWeight(nn.Module):
    def __init__(self, in_channels, out_channels, lw_kernel=3):
        super(ExtResNetBlock_lightWeight, self).__init__()
        # first convolution
        self.conv1 = SingleConv_lightWeight(in_channels, out_channels, lw_kernel=lw_kernel)
        # residual block
        self.conv2 = SingleConv_lightWeight(out_channels, out_channels, lw_kernel=lw_kernel)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual
        self.conv3 = SingleConv_lightWeight(out_channels, out_channels, lw_kernel=lw_kernel)
        self.non_linearity = nn.ELU(inplace=False)

    def forward(self, x):
        # apply first convolution and save the output as a residual
        out = self.conv1(x)
        residual = out
        # residual block
        out = self.conv2(out)
        out = self.conv3(out)

        out = out + residual
        out = self.non_linearity(out)
        return out



class SingleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm='bn', act='relu'):
        super(SingleConv, self).__init__()
        self.add_module('conv', nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
        self.add_module('batchnorm', normalization(out_channels, norm=norm))
        if act == 'relu':
            self.add_module('relu', nn.ReLU(inplace=False))
        elif act == 'lrelu':
            self.add_module('lrelu', nn.LeakyReLU(negative_slope=0.1, inplace=False))
        elif act == 'elu':
            self.add_module('elu', nn.ELU(inplace=False))
        elif act == 'gelu':
            self.add_module('elu', nn.GELU(inplace=False))


class SingleConv_lightWeight(nn.Sequential):
    def __init__(self, in_channels, out_channels, lw_kernel=3, layer_scale_init_value=1e-6):
        super(SingleConv_lightWeight, self).__init__()

        self.dwconv = nn.Conv3d(in_channels, in_channels, kernel_size=lw_kernel, padding=lw_kernel//2, groups=in_channels)
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)
        self.pwconv1 = nn.Linear(in_channels, 2 * in_channels)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(2 * in_channels, out_channels)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_channels)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        if in_channels != out_channels:
            self.skip = nn.Conv3d(in_channels, out_channels, 1)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, D, H, W) -> (N, D, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)  # (N, H, W, C) -> (N, C, H, W)

        x = x + (input if self.in_channels == self.out_channels else self.skip(input))
        return x


class Upsampling(nn.Module):
    def __init__(self, transposed_conv, in_channels=None, out_channels=None, scale_factor=(2, 2, 2), mode='nearest'):
        super(Upsampling, self).__init__()

        if transposed_conv:
            # make sure that the output size reverses the MaxPool3d from the corresponding encoder
            # (D_out = (D_in − 1) ×  stride[0] − 2 ×  padding[0] +  kernel_size[0] +  output_padding[0])
            self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=scale_factor, padding=1)
        else:
            self.upsample = partial(self._interpolate, mode=mode)

    def forward(self, encoder_features, x):
        output_size = encoder_features.size()[2:]
        return self.upsample(x, output_size)

    @staticmethod
    def _interpolate(x, size, mode):
        return F.interpolate(x, size=size, mode=mode)


# Residual 3D UNet
# class ResidualUNet3D(nn.Module):
#     def __init__(self, 
#                 f_maps=[32, 64, 128, 256],   # default
#                 in_channels=1, 
#                 out_channels=13, 
#                 norm="bn",  # default in options
#                 act="relu",  # default in options
#                 use_IP=True,  # Image Pyramid; default in train_bash
#                 use_coord=True,  # Coordinate Convolution; default in train_bash
#                 use_softmax=False,  # my defaults for single class
#                 use_sigmoid=True,  # my defaults for single class
#                 use_lw=False,  # LightWeight; default in train_bash
#                 lw_kernel=3,  # default in train_bash
                
#         ):
#         super(ResidualUNet3D, self).__init__()
        
#         self.use_IP = use_IP
#         self.out_channels = out_channels
#         if self.out_channels > 1:
#             self.use_softmax = use_softmax
#         else:
#             self.use_sigmoid = use_sigmoid
#         self.use_coord = use_coord

#         pool_layer = nn.AvgPool3d

#         if self.use_IP:
#             pools = []
#             for _ in range(len(f_maps) - 1):
#                 pools.append(pool_layer(2))
#             self.pools = nn.ModuleList(pools)

#         # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
#         encoders = []
#         for i, out_feature_num in enumerate(f_maps):
#             if i == 0:
#                 encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, use_IP=False,
#                                   use_coord=self.use_coord,
#                                   pool_layer=pool_layer, norm=norm, act=act,
#                                   use_lw=use_lw, lw_kernel=lw_kernel)
#             else:
#                 # TODO: adapt for anisotropy in the data, i.e. use proper pooling kernel to make the data isotropic after 1-2 pooling operations
#                 encoder = Encoder(f_maps[i - 1], out_feature_num, use_IP=self.use_IP, use_coord=self.use_coord,
#                                   pool_layer=pool_layer, norm=norm, act=act,
#                                   use_lw=use_lw, lw_kernel=lw_kernel)

#             encoders.append(encoder)

#         self.encoders = nn.ModuleList(encoders)


#         # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
#         decoders = []
#         reversed_f_maps = list(reversed(f_maps))
#         for i in range(len(reversed_f_maps) - 1):
#             in_feature_num = reversed_f_maps[i]
#             out_feature_num = reversed_f_maps[i + 1]
#             # TODO: if non-standard pooling was used, make sure to use correct striding for transpose conv
#             # currently strides with a constant stride: (2, 2, 2)
#             decoder = Decoder(in_feature_num, out_feature_num, use_coord=self.use_coord, norm=norm, act=act,
#                               use_lw=use_lw, lw_kernel=lw_kernel)
#             decoders.append(decoder)

#         self.decoders = nn.ModuleList(decoders)


#         self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
#         self.dropout = nn.Dropout3d


#     def forward(self, x):
#         if self.use_IP:
#             img_pyramid = []
#             img_d = x
#             for pool in self.pools:
#                 img_d = pool(img_d)
#                 img_pyramid.append(img_d)

#         encoders_features = []
#         for idx, encoder in enumerate(self.encoders):
#             if self.use_IP and idx > 0:
#                 x = encoder(x, img_pyramid[idx - 1])
#             else:
#                 x = encoder(x)
#             encoders_features.insert(0, x)

#         # remove last
#         encoders_features = encoders_features[1:]


#         for decoder, encoder_features in zip(self.decoders, encoders_features):
#             x = decoder(encoder_features, x)

#         out = self.final_conv(x)

#         if self.out_channels > 1:
#             if self.use_softmax:
#                 out = torch.softmax(out, dim=1)
#         else:
#             if self.use_sigmoid:
#                 out = torch.sigmoid(out)
#         return out