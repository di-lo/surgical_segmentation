# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from monai.networks.blocks import Convolution, UpSample, TransformerBlock, PatchEmbeddingBlock, UnetrBasicBlock, ADN, ResidualUnit, Convolution
from monai.networks.layers.factories import Conv, Pool
from monai.utils import ensure_tuple_rep
from monai.networks.nets.vit import ViT
from sam2.build_sam import build_sam2
from torchvision.models import resnet34
import torch.nn.functional as F


class TwoConv(nn.Sequential):
    """two convolutions."""

    def __init__(
        self,
        dim: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
    ):
        """
        Args:
            dim: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        """
        super().__init__()

        conv_0 = Convolution(dim, in_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        conv_1 = Convolution(dim, out_chns, out_chns, act=act, norm=norm, dropout=dropout, bias=bias, padding=1)
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)


class Down(nn.Sequential):
    """maxpooling downsampling and two convolutions."""

    def __init__(
        self,
        dim: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
    ):
        """
        Args:
            dim: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.

        """
        super().__init__()

        max_pooling = Pool["MAX", dim](kernel_size=2)
        convs = TwoConv(dim, in_chns, out_chns, act, norm, bias, dropout)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)


class UpCat(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    def __init__(
        self,
        dim: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        pre_conv: Optional[Union[nn.Module, str]] = "default",
        interp_mode: str = "linear",
        align_corners: Optional[bool] = True,
        halves: bool = True,
    ):
        """
        Args:
            dim: number of spatial dimensions.
            in_chns: number of input channels to be upsampled.
            cat_chns: number of channels from the decoder.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            bias: whether to have a bias term in convolution blocks.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
            pre_conv: a conv block applied before upsampling.
                Only used in the "nontrainable" or "pixelshuffle" mode.
            interp_mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``}
                Only used in the "nontrainable" mode.
            align_corners: set the align_corners parameter for upsample. Defaults to True.
                Only used in the "nontrainable" mode.
            halves: whether to halve the number of channels during upsampling.
                This parameter does not work on ``nontrainable`` mode if ``pre_conv`` is `None`.
        """
        super().__init__()
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(
            dim,
            in_chns,
            up_chns,
            2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        self.convs = TwoConv(dim, cat_chns + up_chns, out_chns, act, norm, bias, dropout)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]):
        """

        Args:
            x: features to be upsampled.
            x_e: features from the encoder.
        """
        x_0 = self.upsample(x)

        if x_e is not None:
            # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
            dimensions = len(x.shape) - 2
            sp = [0] * (dimensions * 2)
            for i in range(dimensions):
                if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                    sp[i * 2 + 1] = 1
            x_0 = torch.nn.functional.pad(x_0, sp, "replicate")

            x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        else:
            x = self.convs(x_0)

        return x


class mid_vit(nn.Module):
    def __init__(
            self,
            dim: int,
            in_channels,
            img_size,
            patch_size,
            hidden_size,
            mlp_dim,
            num_heads,
            pos_embed,
            dropout: Union[float, tuple] = 0.0,
    ):
        super().__init__()

        self.num_layers = 2
        img_size = ensure_tuple_rep(img_size, dim)
        self.patch_size = ensure_tuple_rep(patch_size, dim)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size

        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed_type=pos_embed,
            classification=False,
            dropout_rate=dropout,
            spatial_dims=dim,
        )

        self.up = UpSample(
            dim,
            hidden_size,
            in_channels,
            [8, 4, 2],
            mode='deconv',
            pre_conv='default',
            interp_mode='linear',
            align_corners=True,
        )

    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)
        x = x.view(new_view)
        new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, x):
        x1, _ = self.vit(x)
        x2 = self.proj_feat(x1, self.hidden_size, self.feat_size)
        x3 = self.up(x2)
        # print(x3.shape)
        return x3


class cats(nn.Module):
    def __init__(
        self,
        dimensions: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Union[str, tuple] = ("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        image_size: tuple = (96, 96, 96),
        # patch_size: tuple = (8, 4, 2),
        # hidden_size: int = 768,
        # feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "sincos", # perceptron
        device: str = 'cpu',
        sam2_checkpoint: bool = False,
        cnn_checkpoint: bool = False,
        sam2_pth: str = None,
    ):

        super().__init__()
        self.dimensions = dimensions

        fea = ensure_tuple_rep(features, 6)
        print(f"cats features: {fea}.")

        self.conv_0 = TwoConv(dimensions, in_channels, features[0], act, norm, bias, dropout)
        self.down_0 = Down(dimensions, fea[0], fea[0], act, norm, bias, dropout)
        self.down_1 = Down(dimensions, fea[0], fea[1], act, norm, bias, dropout)
        self.down_2 = Down(dimensions, fea[1], fea[2], act, norm, bias, dropout)
        self.down_3 = Down(dimensions, fea[2], fea[3], act, norm, bias, dropout)
        self.down_4 = Down(dimensions, fea[3], fea[4], act, norm, bias, dropout)

        self.upcat_4 = UpCat(dimensions, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample)
        self.upcat_3 = UpCat(dimensions, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample)
        self.upcat_2 = UpCat(dimensions, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample)
        self.upcat_1 = UpCat(dimensions, fea[1], fea[0], fea[5], act, norm, bias, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", dimensions](fea[5], out_channels, kernel_size=1)

        self.num_layers = 12
        img_size = ensure_tuple_rep(image_size, dimensions)
        self.patch_size = ensure_tuple_rep(16, dimensions)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False

        if sam2_checkpoint:
            sam2_model = build_sam2(config_file='sam2_hiera_l.yaml', ckpt_path=sam2_pth, device=device)
            self.vit = sam2_model.image_encoder.trunk
            self.sam2_proj4 = nn.Sequential(
                nn.Conv2d(1152, 256, kernel_size=1),
                nn.Upsample(size=(16, 16), mode='bilinear', align_corners=False)
            )
            self.sam2_proj3 = nn.Sequential(
                nn.Conv2d(576, 128, kernel_size=1),
                nn.Upsample(size=(32, 32), mode='bilinear', align_corners=False)
            )
            self.sam2_proj2 = nn.Sequential(
                nn.Conv2d(288, 64, kernel_size=1),
                nn.Upsample(size=(64, 64), mode='bilinear', align_corners=False)
            )
            self.sam2_proj1 = nn.Sequential(
                nn.Conv2d(144, 32, kernel_size=1),
                nn.Upsample(size=(128, 128), mode='bilinear', align_corners=False)
            )
        else:
            self.vit = ViT(
                in_channels=in_channels,
                img_size=img_size,
                patch_size=self.patch_size,
                hidden_size=self.hidden_size,
                mlp_dim=mlp_dim,
                num_layers=self.num_layers,
                num_heads=num_heads,
                pos_embed_type=pos_embed,
                classification=False,
                dropout_rate=0,
                spatial_dims=dimensions,
            )

        if cnn_checkpoint:
            # Load ResNet34 pretrained
            resnet = resnet34(pretrained=True)

            # CNN encoder blocks
            self.encoder_cnn = nn.ModuleDict({
                "enc0": nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu),  # 64 x 112 x 112
                "enc1": nn.Sequential(resnet.maxpool, resnet.layer1),  # 64 x 56 x 56
                "enc2": resnet.layer2,  # 128 x 28 x 28
                "enc3": resnet.layer3,  # 256 x 14 x 14
                "enc4": resnet.layer4  # 512 x 7 x 7
            })
            self.cnn_proj0 = nn.Conv2d(64, 32, kernel_size=1)
            self.cnn_proj1 = nn.Conv2d(64, 32, kernel_size=1)
            self.cnn_proj2 = nn.Conv2d(128, 64, kernel_size=1)
            self.cnn_proj3 = nn.Conv2d(256, 128, kernel_size=1)
            self.cnn_proj4 = nn.Conv2d(512, 256, kernel_size=1)


        # self.bottom_conv = Conv["conv", dimensions](self.hidden_size, fea[4], kernel_size=3, padding=1)
        self.bottom_conv = ResidualUnit(dimensions, self.hidden_size, fea[4],
                                        strides=1, kernel_size=3, subunits=2,
                                        adn_ordering='NDA', act='RELU', norm='batch')


        self.up3 = UpSample(
            dimensions,
            hidden_size,
            fea[3],
            2,
            mode='deconv',
            pre_conv='default',
            interp_mode='linear',
            align_corners=True,
        )

        self.fourth_conv = ResidualUnit(dimensions, fea[3], fea[3],
                                        strides=1, kernel_size=3, subunits=2,
                                        adn_ordering='NDA', act='RELU', norm='batch')


        self.up21 = UpSample(
            dimensions,
            hidden_size,
            fea[2],
            2,
            mode='deconv',
            pre_conv='default',
            interp_mode='linear',
            align_corners=True,
        )
        self.up22 = UpSample(
            dimensions,
            fea[2],
            fea[2],
            2,
            mode='deconv',
            pre_conv='default',
            interp_mode='linear',
            align_corners=True,
        )
        self.third_conv1 = ResidualUnit(dimensions, fea[2], fea[2],
                                        strides=1, kernel_size=3, subunits=2,
                                        adn_ordering='NDA', act='RELU', norm='batch')
        self.third_conv2 = ResidualUnit(dimensions, fea[2], fea[2],
                                       strides=1, kernel_size=3, subunits=2,
                                       adn_ordering='NDA', act='RELU', norm='batch')


        self.up11 = UpSample(
            dimensions,
            hidden_size,
            fea[1],
            2,
            mode='deconv',
            pre_conv='default',
            interp_mode='linear',
            align_corners=True,
        )
        self.up12 = UpSample(
            dimensions,
            fea[1],
            fea[1],
            2,
            mode='deconv',
            pre_conv='default',
            interp_mode='linear',
            align_corners=True,
        )
        self.up13 = UpSample(
            dimensions,
            fea[1],
            fea[1],
            2,
            mode='deconv',
            pre_conv='default',
            interp_mode='linear',
            align_corners=True,
        )
        self.second_conv1 = ResidualUnit(dimensions, fea[1], fea[1],
                                        strides=1, kernel_size=3, subunits=2,
                                        adn_ordering='NDA', act='RELU', norm='batch')
        self.second_conv2 = ResidualUnit(dimensions, fea[1], fea[1],
                                         strides=1, kernel_size=3, subunits=2,
                                         adn_ordering='NDA', act='RELU', norm='batch')
        self.second_conv3 = ResidualUnit(dimensions, fea[1], fea[1],
                                         strides=1, kernel_size=3, subunits=2,
                                         adn_ordering='NDA', act='RELU', norm='batch')

    def proj_feat(self, x, hidden_size, feat_size):
        new_view = (x.size(0), *feat_size, hidden_size)  # [B, H, W, C]
        x = x.view(new_view)
        if self.dimensions == 3:
            new_axes = (0, len(x.shape) - 1) + tuple(d + 1 for d in range(len(feat_size)))  # [B, C, D, H, W]
        else:
            new_axes = (0, 3, 1, 2)  # [B, C, H, W]
        x = x.permute(new_axes).contiguous()
        return x

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``, N is defined by `dimensions`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.

        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.
        """
        if hasattr(self, "encoder_cnn"):
            cnn_x0 = self.encoder_cnn["enc0"](x)  # [B, 64, 112, 112]
            cnn_x1 = self.encoder_cnn["enc1"](cnn_x0)  # [B, 64, 56, 56]
            cnn_x2 = self.encoder_cnn["enc2"](cnn_x1)  # [B, 128, 28, 28]
            cnn_x3 = self.encoder_cnn["enc3"](cnn_x2)  # [B, 256, 14, 14]
            cnn_x4 = self.encoder_cnn["enc4"](cnn_x3)  # [B, 512, 7, 7]

            x0 = self.conv_0(x)
            x0 = self.down_0(x0)
            x1 = self.down_1(x0)
            x2 = self.down_2(x1)
            x3 = self.down_3(x2)
            x4 = self.down_4(x3)

            # print(x1.shape, self.cnn_proj1(cnn_x1).shape)
            x1 = x1 + self.cnn_proj1(cnn_x1)  # 56×56
            x2 = x2 + self.cnn_proj2(cnn_x2)  # 28×28
            x3 = x3 + self.cnn_proj3(cnn_x3)  # 14×14
            x4 = x4 + self.cnn_proj4(cnn_x4)  # 7×7
        else:
            x0 = self.conv_0(x)
            x1 = self.down_1(x0)
            x2 = self.down_2(x1)
            x3 = self.down_3(x2)
            x4 = self.down_4(x3)

        if isinstance(self.vit, ViT):
            # bottom
            xt_4, hidden_states_out = self.vit(x)
            xt_4 = self.bottom_conv(self.proj_feat(xt_4, self.hidden_size, self.feat_size))

            # 1/16 -> 1/8
            xt_3 = hidden_states_out[9]
            xt_3 = (self.proj_feat(xt_3, self.hidden_size, self.feat_size))
            xt_3 = self.up3(xt_3)
            xt_3 = self.fourth_conv(xt_3)

            # 1/16 -> 1/4
            xt_2 = hidden_states_out[6]
            xt_2 = self.up21(self.proj_feat(xt_2, self.hidden_size, self.feat_size))
            # xt_2 = self.third_conv1(xt_2)
            xt_2 = self.up22(xt_2)
            xt_2 = self.third_conv1(xt_2)

            # 1/16 -> 1/2
            xt_1 = hidden_states_out[3]
            xt_1 = self.proj_feat(xt_1, self.hidden_size, self.feat_size)
            xt_1 = self.up11(xt_1)
            # xt_1 = self.second_conv1(xt_1)
            xt_1 = self.up12(xt_1)
            # xt_1 = self.second_conv2(xt_1)
            xt_1 = self.up13(xt_1)
            xt_1 = self.second_conv1(xt_1)
        else:
            x1_sam2, x2_sam2, x3_sam2, x4_sam2 = self.vit(x)

            xt_4 = self.sam2_proj4(x4_sam2)
            xt_3 = self.sam2_proj3(x3_sam2)
            xt_2 = self.sam2_proj2(x2_sam2)
            xt_1 = self.sam2_proj1(x1_sam2)

        x4 = xt_4 + x4
        x3 = xt_3 + x3
        x2 = xt_2 + x2
        x1 = xt_1 + x1

        u4 = self.upcat_4(x4, x3)
        u3 = self.upcat_3(u4, x2)
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x0)

        logits = self.final_conv(u1)
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        return logits


