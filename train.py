import random

import torch
from fontTools.misc.cython import returns
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from config.config_args import *
from config.config_setup import get_net, get_dataset, init_seeds, get_optimizer_and_scheduler
from util.utils import *
import os
import sys
import torch.nn.functional as F
sys.path = list(dict.fromkeys(sys.path))
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
project_root = os.path.dirname(os.getcwd())
if project_root in sys.path:
    sys.path.remove(project_root)

def train_net_sup(args, net, trainset, valset, save_cp=True):
    n_val, n_train = len(valset), len(trainset)
    logging.info("total frames is: {}".format(n_train))

    def worker_init_fn(worker_id):
        random.seed(42 + worker_id)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, pin_memory=True,
                                      num_workers=args.num_workers, shuffle=True, drop_last=True, worker_init_fn=worker_init_fn)

    # train_loader_batch = next(iter(train_loader))
    # print(train_loader_batch)

    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    # val_loader_batch = next(iter(val_loader))
    # print(val_loader_batch)

    logging.info(f'''Starting training:
        Epochs:          {args.total_epoch}
        Batch size:      {args.batch_size}
        Learning rate:   {args.lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu').type}
    ''')

    optimizer, scheduler = get_optimizer_and_scheduler(args, net)


    criterion = CrossEntropyLoss() #CHANGED FROM BCE

    best_dice1 = 0
    for epoch in range(args.total_epoch):
        train_sup(args, train_loader, net, criterion, optimizer, epoch, scheduler)
        mean_dice, std_dice, dice_per_class = validate_sup(net, val_loader, args.device)

        logging.info('')
        logging.info('Model, batch-wise validation Dice coeff: {}, std: {}, Dice per class: {}'.format(mean_dice, std_dice, dice_per_class))
        logging.info('===================================================================================')


        if save_cp and mean_dice > best_dice1:
            save_checkpoint(net, args.save_dir, epoch, best=True)
            best_dice1 = mean_dice

        torch.cuda.empty_cache()

def validate_sup(net, loader, device):
    dice_list = []
    net.eval()
    with torch.no_grad():
        with tqdm(total=len(loader), desc='Validation round', unit='batch', leave=False) as pbar:
            for i, (batch) in enumerate(loader):
                input, target = batch['image'].to(device=device, dtype=torch.float32), batch['label'].to(device=device, dtype=torch.long)
                if args.net == 'cats2d':
                    output, _, _ = net(input)
                elif args.net == 'segformer':
                    output = net(pixel_values=batch['image'].to(device=device)).logits
                    output = F.interpolate(
                        output,
                        size=target.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                elif args.net == 'deeplab':
                    # output = net.model(input)['out']
                    output = net.model(input)
                    output = F.interpolate(output, size=target.shape[-2:], mode='bilinear', align_corners=False)

                elif args.net == 'mask2former':
                    outputs = net(pixel_values=input)
                    class_probs = outputs.class_queries_logits.softmax(dim=-1)[..., :-1]
                    mask_logits = outputs.masks_queries_logits
                    output = torch.einsum("bqhw,bqc->bchw", mask_logits, class_probs)
                    output = F.interpolate(output, size=target.shape[-2:], mode="bilinear", align_corners=False)
                elif args.net == 'swinunet' or args.net == 'cats':
                    output = net(input)
                    output = F.interpolate(output, size=target.shape[-2:], mode='bilinear', align_corners=False)
                else:
                    output = net(input) #unet and dscnet

                output_prob = F.softmax(output, dim=1)
                output_pred = torch.argmax(output_prob, dim=1)  # shape: (B, H, W)

                output_pred = output_pred.detach().cpu()
                target = target.detach().cpu()

                mean_dice, std_dice, dice_per_class = dice_coeff(output_pred, target, num_classes=5) #dice_per_class is for the last image in the batch
                dice_list.append(mean_dice)

                pbar.update(1)
            pbar.close()
        return np.mean(dice_list), np.std(dice_list), dice_per_class


def train_sup(args, train_loader, model, criterion, optimizer, epoch, scheduler):
    model.train()
    loss1_list_sup = []

    pbar = tqdm(total=len(train_loader))

    for batch_labeled in train_loader:
        input_labeled = batch_labeled['image'].to(device=args.device, dtype=torch.float32)
        # target_labeled = batch_labeled['label'].to(device=args.device)
        # target_labeled = target_labeled.long()
        target_labeled = batch_labeled['label'].to(device=args.device, dtype=torch.long)
        target_labeled = target_labeled.squeeze(1)

        if args.net == 'cats2d':
            output1, _, _ = model(input_labeled)
        elif args.net == 'segformer':
            output1 = model(input_labeled).logits
            output1 = F.interpolate(
                output1,
                size=target_labeled.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        elif args.net == 'deeplab':
            # output1 = model.model(input_labeled)['out']
            output1 = model.model(input_labeled)
            output1 = F.interpolate(output1, size=target_labeled.shape[-2:], mode='bilinear', align_corners=False)
        elif args.net == 'mask2former':

            outputs = model(pixel_values=input_labeled)

            class_probs = outputs.class_queries_logits.softmax(dim=-1)[..., :-1]
            mask_logits = outputs.masks_queries_logits  # [B, Q, H, W]

            output1 = torch.einsum("bqhw,bqc->bchw", mask_logits, class_probs)
            output1 = F.interpolate(output1, size=target_labeled.shape[-2:], mode="bilinear", align_corners=False)
        elif args.net == 'swinunet' or args.net == 'cats':
            output1 = model(input_labeled)
            output1 = F.interpolate(
                output1,
                size=target_labeled.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        else:
            output1 = model(input_labeled) #unet and dscnet

        loss = criterion(output1, target_labeled)

        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_value_(net.parameters(), 0.1)
        optimizer.step()
        loss1_list_sup.append(loss.item())
        pbar.update(1)

    logging.info('===================================================================================')
    logging.info('Epoch: {}, model1 supervised loss: {}'.format(epoch, np.mean(loss1_list_sup)))


    pbar.close()

    checkpoint_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'mean loss': np.mean(loss1_list_sup),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    save_checkpoint(model, net_dict=checkpoint_dict, save_dir=args.save_dir, epoch=epoch, best=False)
    logging.info(f"Epoch {epoch + 1}, learning rate: {scheduler.get_last_lr()}")
    scheduler.step()


if __name__ == '__main__':
    init_seeds(42)
    args = parser.parse_args()
    setup_logging(args, mode='train')
    logging.info(os.path.dirname(os.path.abspath(__file__)))
    logging.info(args)

    assert args.json_path is not None, 'input your split file'

    train_set = get_dataset(args, mode='train', json=True)
    val_set = get_dataset(args, mode='val', json=True)

    net = get_net(args, net=args.net)

    logging.info('Models and datasets are loaded')

    logging.info('Training CAO full supervision...')
    train_net_sup(args, net=net, trainset=train_set, valset=val_set)
    print("done!")
