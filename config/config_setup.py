import cv2
import segmentation_models_pytorch as smp
from transformers import SegformerConfig, SegformerForSemanticSegmentation, PretrainedConfig
import albumentations as A
from albumentations.core.composition import Compose
from src.dataset.CAO_dataset import CAO_dataset
import random, os
import numpy as np
import logging
from torch import optim
from src.models.CATS2d import CATS2d
from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation
import torch.nn as nn
import torch
from src.models.DSCNet import DSCNet

def get_net(args, pretrain=False, model=None, net=None):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple MPS (Mac M1/M2)
    else:
        device = torch.device("cpu")

    net = args.net
    logging.info(f'Using device {device}')
    logging.info(f'Building:  {net}')

    if net == 'unet':
        if args.weights:
            net = smp.Unet(encoder_name="resnet34", encoder_weights='imagenet',
                       in_channels=args.in_channels, classes=args.out_channels)
        else:
            net = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                       in_channels=args.in_channels, classes=args.out_channels)

    elif net == 'deeplabplus':
        if args.weights:
            net = smp.DeepLabV3Plus(encoder_name='resnet34', encoder_weights='imagenet',
                                in_channels=args.in_channels, classes=args.out_channels)
        else:
            net = smp.DeepLabV3Plus(encoder_name='resnet34', encoder_weights=None,
                                in_channels=args.in_channels, classes=args.out_channels)

    elif net == 'cats2d':
        # pretrained weight: args.sam2_path
        if args.weights:
            net = CATS2d(args.sam2_path, device=device)
        else:
            net = CATS2d(device=device)

    elif net == 'deeplab':

        if args.deeplab_ident == 'mobilenet':
            from qai_hub_models.models.deeplabv3_plus_mobilenet import Model as QualcommDeepLab
            model = QualcommDeepLab.from_pretrained()
            in_ch = model.model.decoder.last_conv[8].in_channels
            model.model.decoder.last_conv[8] = nn.Conv2d(in_ch, args.out_channels, kernel_size=1)

        if args.deeplab_ident == 'resnet50':
            from qai_hub_models.models.deeplabv3_resnet50 import Model as QualcommDeepLab
            model = QualcommDeepLab.from_pretrained()
            in_ch = model.model.classifier[4].in_channels
            model.model.classifier[4] = nn.Conv2d(in_ch, args.out_channels, kernel_size=1)

            in_ch2 = model.model.aux_classifier[4].in_channels
            model.model.aux_classifier[4] = nn.Conv2d(in_ch2, args.out_channels, kernel_size=1)

        net = model.to(device)

    elif net == 'segformer':

        if args.weights:
            net = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b2-finetuned-ade-512-512",
                num_labels=args.out_channels,
                ignore_mismatched_sizes=True
            ).to(device)
        else:
            config = SegformerConfig(num_labels = args.out_channels)
            net = SegformerForSemanticSegmentation(config).to(device)

    elif args.net == "mask2former":

        if args.weights:
            net = Mask2FormerForUniversalSegmentation.from_pretrained(
                "facebook/mask2former-swin-large-cityscapes-semantic",
                num_labels=args.out_channels,
                ignore_mismatched_sizes=True
            ).to(device)
        else:
            custom_dict = {
                "num_labels": args.out_channels,
                "use_pretrained_backbone": False,
            }
            config = Mask2FormerConfig(**custom_dict)
            net = Mask2FormerForUniversalSegmentation(config).to(device)


    elif args.net == 'dscnet':
        net = DSCNet(
            n_channels=args.in_channels,
            n_classes=args.out_channels,
            kernel_size=9,
            extend_scope=1,
            if_offset=True,
            device=device,
            number=12,
            dim=1,
        ).to(device)

    elif net == 'swinunet':
        from src.models.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
        import os

        # Create model
        model = SwinTransformerSys(
            img_size=448,
            patch_size=4,
            in_chans=args.in_channels,
            num_classes=args.out_channels,
            embed_dim=96,
            depths=[2, 2, 2, 2],
            depths_decoder=[1, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            final_upsample="expand_first"
        )

        net = model.to(device)

        if os.path.isfile(args.swinunet_path):
            print(f"Loading Swin encoder from {args.swinunet_path}")
            checkpoint = torch.load(args.swinunet_path, map_location=device)
            pretrained_dict = checkpoint.get('model', checkpoint)
            model_dict = model.state_dict()

            # Filter out attention masks
            filtered_dict = {
                k: v for k, v in pretrained_dict.items()
                if k in model_dict and v.size() == model_dict[k].size() and "attn_mask" not in k
            }
            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict, strict=False)

            print(f"Loaded {len(filtered_dict)} layers from pretrained Swin encoder.")
        else:
            print("Pretrained Swin encoder not found, training from scratch.")

    elif net == 'cats':
        from src.models.cats import cats
        if args.weights:
            net = cats(
                dimensions=2,
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                image_size=(args.height, args.width),
                device=str(device),
                sam2_checkpoint= True,
                cnn_checkpoint= True,
                sam2_pth= args.sam2_path,
            ).to(device)
        else:
            net = cats(
                dimensions=2,
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                image_size=(args.height, args.width),
                device=str(device),
                sam2_checkpoint= False,
                cnn_checkpoint= False,
            ).to(device)

    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    logging.info(f"Total trainable parameters: {total_params}")

    if pretrain and args.weights:
        import os
        pretrain_path = os.path.join(args.base_dir, args.name, 'cp', 'best_net.pth')
        net.load_state_dict(torch.load(pretrain_path, map_location='cuda:0'), strict=False)
        logging.info(f'Model {model}  loaded from {pretrain_path}')

    net.to(device=device)
    return net


def get_optimizer_and_scheduler(args, net):
    params = filter(lambda p: p.requires_grad, net.parameters())  # added from
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_epoch, eta_min=args.min_lr)

    return optimizer, scheduler


def get_dataset(args, mode=None, json=False):
    if mode is None:
        raise ValueError('mode must be specified')

    if args.net == 'segformer' and mode != 'train':
        transform = Compose([
            A.Resize(args.height, args.width, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ], seed=42)
    elif mode == 'train':
        transform = Compose([
            A.Resize(args.height, args.width),
            A.RandomRotate90(),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine((0.7, 1.3), {'x': (-0.3, 0.3), 'y': (-0.3, 0.3)}, rotate=(-360, 360), p=0.25),
            A.GaussianBlur(p=0.1),
            A.AutoContrast(p=0.1),
            A.MedianBlur(blur_limit=15, p=0.1),
            A.RandomGamma(p=0.1),
            A.Defocus(p=0.1),
            A.RandomFog(alpha_coef=0.1, p=0.1),
            A.RandomBrightnessContrast(p=0.1),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ], seed=42)
    else:
        transform = Compose([
            A.Resize(args.height, args.width),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                             ], seed=42)


    dataset_cao = CAO_dataset(args, mode=mode, transform=transform)

    return dataset_cao


def init_seeds(seed=42, cuda_deterministic=True):
    from torch.backends import cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='internimage')
    parser.add_argument('--device', type=str, default=0)
    args = parser.parse_args()
    get_net(args)
