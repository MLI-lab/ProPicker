import torch
import torch.nn as nn
from functools import partial

from .layers import AddCoords, ExtResNetBlock, ExtResNetBlock_lightWeight, Upsampling


class ResidualUNet3DDecoder(nn.Module):
    def __init__(self, 
                f_maps=[32, 64, 128, 256],   # default
                out_channels=1, 
                norm="bn",  # default in options
                act="relu",  # default in options
                use_coord=True,  # Coordinate Convolution; default in train_bash
                use_softmax=False,  # my defaults for single class
                use_sigmoid=True,  # my defaults for single class
                use_lw=False,  # LightWeight; default in train_bash
                lw_kernel=3,  # default in train_bash
                promptable=True,  # my modification
                prompt_dim=32,  # my modification
                film_activation=None,  # my modification
        ):
        super().__init__()
        
        self.out_channels = out_channels
        if self.out_channels > 1:
            self.use_softmax = use_softmax
        else:
            self.use_sigmoid = use_sigmoid
        self.use_coord = use_coord


        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i]
            out_feature_num = reversed_f_maps[i + 1]
            # TODO: if non-standard pooling was used, make sure to use correct striding for transpose conv
            # currently strides with a constant stride: (2, 2, 2)
            decoder = Decoder(in_feature_num, out_feature_num, use_coord=self.use_coord, norm=norm, act=act,
                              use_lw=use_lw, lw_kernel=lw_kernel, promptable=promptable, prompt_dim=prompt_dim, film_activation=film_activation)
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        self.dropout = nn.Dropout3d


    def forward(self, 
                x,  # x is last encoder feature
                cats,  # cats is encoder_features
                prompt=None,
        ):
        for decoder, cat in zip(self.decoders, cats):
            x = decoder(cat, x, prompt=prompt)

        out = self.final_conv(x)

        if self.out_channels > 1:
            if self.use_softmax:
                out = torch.softmax(out, dim=1)
        else:
            if self.use_sigmoid:
                out = torch.sigmoid(out)
        return out
    
    
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=(2, 2, 2), mode='nearest',
                 padding=1, use_coord=False, norm='bn', act='relu', 
                 use_lw=False, lw_kernel=3, promptable=True, prompt_dim=32, film_activation=None):
        super(Decoder, self).__init__()
        self.use_coord = use_coord
        if self.use_coord:
            self.coord_conv = AddCoords(rank=3, with_r=False)

        # if basic_module=ExtResNetBlock use transposed convolution upsampling and summation joining
        self.upsampling = Upsampling(transposed_conv=True, in_channels=in_channels, out_channels=out_channels,
                                     scale_factor=scale_factor, mode=mode)
        # sum joining
        self.joining = partial(self._joining, concat=False)
        # adapt the number of in_channels for the ExtResNetBlock
        in_channels = out_channels + 3 if self.use_coord else out_channels

        if use_lw:
            if promptable:
                raise NotImplementedError("LightWeight and promptable not implemented together")
            self.basic_module = ExtResNetBlock_lightWeight(in_channels, out_channels, lw_kernel=lw_kernel)
        else:
            self.basic_module = ExtResNetBlock(in_channels, out_channels, norm=norm, act=act, promptable=promptable, prompt_dim=prompt_dim, film_activation=film_activation)


    def forward(self, encoder_features, x, ReturnInput=False, prompt=None):
        x = self.upsampling(encoder_features, x)
        x = self.joining(encoder_features, x)
        if self.use_coord:
            x = self.coord_conv(x)
        if ReturnInput:
            x1 = self.basic_module(x, prompt=prompt)
            return x1, x
        x = self.basic_module(x, prompt=prompt)
        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x