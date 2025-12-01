import torch
from torch import nn
from .layers import AddCoords, ExtResNetBlock, ExtResNetBlock_lightWeight

class ResidualUNet3DEncoder(nn.Module):
    def __init__(self, 
                f_maps=[32, 64, 128, 256],   # default
                in_channels=1, 
                norm="bn",  # default in options
                act="relu",  # default in options
                use_IP=True,  # Image Pyramid; default in train_bash
                use_coord=True,  # Coordinate Convolution; default in train_bash
                use_lw=False,  # LightWeight; default in train_bash
                lw_kernel=3,  # default in train_bash
                
        ):
        super().__init__()
        
        self.use_IP = use_IP
        self.use_coord = use_coord

        pool_layer = nn.AvgPool3d

        if self.use_IP:
            pools = []
            for _ in range(len(f_maps) - 1):
                pools.append(pool_layer(2))
            self.pools = nn.ModuleList(pools)

        # create encoder path consisting of Encoder modules. Depth of the encoder is equal to `len(f_maps)`
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, use_IP=False,
                                  use_coord=self.use_coord,
                                  pool_layer=pool_layer, norm=norm, act=act,
                                  use_lw=use_lw, lw_kernel=lw_kernel)
            else:
                # TODO: adapt for anisotropy in the data, i.e. use proper pooling kernel to make the data isotropic after 1-2 pooling operations
                encoder = Encoder(f_maps[i - 1], out_feature_num, use_IP=self.use_IP, use_coord=self.use_coord,
                                  pool_layer=pool_layer, norm=norm, act=act,
                                  use_lw=use_lw, lw_kernel=lw_kernel)

            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)



    def forward(self, x):
        if self.use_IP:
            img_pyramid = []
            img_d = x
            for pool in self.pools:
                img_d = pool(img_d)
                img_pyramid.append(img_d)

        encoders_features = []
        for idx, encoder in enumerate(self.encoders):
            if self.use_IP and idx > 0:
                x = encoder(x, img_pyramid[idx - 1])
            else:
                x = encoder(x)
            encoders_features.insert(0, x)

        # remove last
        #encoders_features = encoders_features[1:]
        return None, encoders_features


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, apply_pooling=True, use_IP=False, use_coord=False,
                 pool_layer=nn.MaxPool3d, norm='bn', act='relu',
                 use_lw=False, lw_kernel=3, input_channels=1):
        super(Encoder, self).__init__()
        if apply_pooling:
            self.pooling = pool_layer(kernel_size=2)
        else:
            self.pooling = None

        self.use_IP = use_IP
        self.use_coord = use_coord
        inplaces = in_channels + input_channels if self.use_IP else in_channels
        inplaces = inplaces + 3 if self.use_coord else inplaces

        if use_lw:
            self.basic_module = ExtResNetBlock_lightWeight(inplaces, out_channels, lw_kernel=lw_kernel)
        else:
            self.basic_module = ExtResNetBlock(inplaces, out_channels, norm=norm, act=act)
        if self.use_coord:
            self.coord_conv = AddCoords(rank=3, with_r=False)

    def forward(self, x, scaled_img=None):
        if self.pooling is not None:
            x = self.pooling(x)
        if self.use_IP:
            x = torch.cat([x, scaled_img], dim=1)
        if self.use_coord:
            x = self.coord_conv(x)
        x = self.basic_module(x)
        return x
