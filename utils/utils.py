import os, torch, logging
from torch.autograd import Function
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from src.config.config_args import *

def save_checkpoint(net, save_dir, epoch, net_dict=None, best=False):
    os.makedirs(save_dir, exist_ok=True)
    save_checkpoint_dir = os.path.join(save_dir, 'cp')
    os.makedirs(save_checkpoint_dir, exist_ok=True)

    if best:
        torch.save(net.state_dict(), os.path.join(save_checkpoint_dir, 'best_net.pth'))
        logging.info(f'best Checkpoint net {epoch + 1} saved !')
    else:
        torch.save(net_dict, os.path.join(save_checkpoint_dir, 'last_net.pth'))
        logging.info(f'last Checkpoint net {epoch + 1} saved !')

        if epoch == 0 or epoch % 10 == 9:
            filename = f'epoch{epoch}.pth'
            torch.save(net_dict, os.path.join(save_checkpoint_dir, filename))
            logging.info(f'Epoch checkpoint {filename} saved !')


def dice_coefficient_multiclass_batch(preds, targets, num_classes, epsilon=1e-6):

    preds = preds.squeeze(1)
    targets = targets.squeeze(1)
    preds_one_hot = torch.nn.functional.one_hot(preds, num_classes=num_classes).permute(0, 3, 1, 2)
    targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2)

    preds_flat = preds_one_hot.view(preds_one_hot.shape[0], num_classes, -1)
    targets_flat = targets_one_hot.view(targets_one_hot.shape[0], num_classes, -1)

    intersection = torch.sum(preds_flat * targets_flat, dim=2)
    union = torch.sum(preds_flat, dim=2) + torch.sum(targets_flat, dim=2)

    dice = (2. * intersection + epsilon) / (union + epsilon)

    return dice


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.00001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(preds, targets, num_classes, epsilon=1e-6):
    """
    It will calculate the average dice score across all classes for a given prediction and ground truth pair.
    It will stack those dice scores across all pairs and taking the average as a return value.

    Args:
        preds (Tensor): shape (B, H, W), predicted class indices.
        targets (Tensor): shape (B, H, W), ground truth class indices.
        num_classes (int): number of classes.
        epsilon (float): small constant to avoid division by zero.

    Returns:
        Tuple[float, float]: mean and std of Dice scores across the batch.
    """
    batch_dice = []

    for pred, target in zip(preds, targets):
        # print(torch.unique(pred), torch.unique(target))

        dice_per_class = []
        for cls in range(1, num_classes):
            pred_cls = (pred == cls).float()
            target_cls = (target == cls).float()

            if pred_cls.sum() == 0 and target_cls.sum() == 0:
                dice = torch.tensor(1.0)
            else:
                intersection = (pred_cls * target_cls).sum()
                union = pred_cls.sum() + target_cls.sum()
                dice = (2. * intersection + epsilon) / (union + epsilon)

            dice_per_class.append(dice)

        mean_dice = torch.nanmean(torch.stack(dice_per_class))
        batch_dice.append(mean_dice)

    batch_dice_tensor = torch.stack(batch_dice)
    return batch_dice_tensor.mean().item(), batch_dice_tensor.std().item(), dice_per_class


