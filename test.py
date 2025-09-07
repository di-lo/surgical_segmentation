import os.path
import shutil

from dataset.CAO_dataset import CAO_dataset
from torch.utils.data import DataLoader
from albumentations.core.composition import Compose
from albumentations import Resize
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from util.utils import dice_coefficient_multiclass_batch, dice_coeff
from config.config_args import *
from config.config_setup import get_net, init_seeds
from PIL import Image
from config.config_setup import get_dataset
import cv2
import os
import sys
sys.path = list(dict.fromkeys(sys.path))
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
project_root = os.path.dirname(os.getcwd())
if project_root in sys.path:
    sys.path.remove(project_root)

import numpy as np
import torch

import cv2

args = parser.parse_args()
if args.classChange:
    CLASS_COLORS = {
        0: (0, 0, 0),           # Background - black
        1: (0, 0, 255),         # Class 1 - blue
        2: (0, 255, 0),         # Class 2 - green
        4: (255, 0, 0),         # Class 3 - red
        3: (0, 255, 255),       # Class 4 - cyan
    }
else:
    CLASS_COLORS = {
        0: (0, 0, 0),           # Background - black
        1: (0, 0, 255),         # Class 1 - blue
        2: (0, 255, 0),         # Class 2 - green
        3: (255, 0, 0),         # Class 3 - red
        4: (0, 255, 255),       # Class 4 - cyan
    }

def colorize_mask(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        color_mask[mask == class_id] = color
    return color_mask

from PIL import Image

def save_overlay_prediction(original_image_path, pred_mask_tensor, save_path, alpha=0.5, cropped=False):
    # Load and prepare original image
    image = np.array(Image.open(original_image_path).convert("RGB"))
    if cropped:
        image = image[:, 420:-420, :]

    # Prepare predicted mask
    pred_mask = pred_mask_tensor.cpu().numpy().astype(np.uint8)
    if pred_mask.shape != image.shape[:2]:
        pred_mask = cv2.resize(pred_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Generate colored mask (RGB)
    color_mask = colorize_mask(pred_mask)

    # Create an alpha mask where class != 0 → visible, class == 0 → transparent
    alpha_mask = np.where(pred_mask != 0, int(alpha * 255), 0).astype(np.uint8)

    # Stack RGB + Alpha → RGBA
    color_mask_rgba = np.dstack((color_mask, alpha_mask))  # Shape: (H, W, 4)

    # Convert to PIL Images
    overlay = Image.fromarray(color_mask_rgba, mode="RGBA")
    background = Image.fromarray(image, mode="RGB").convert("RGBA")

    # Alpha-composite (overlay on top of original)
    result = Image.alpha_composite(background, overlay)

    # Save result
    result.save(save_path)

def validate_baseline(args, net, loader, save_results_dir=None):
    if args.save_results and not os.path.exists(save_results_dir):
        os.makedirs(save_results_dir, exist_ok=True)

    net.eval()
    dice_records = []
    per_image_dice = []
    csv_rows = [] # [img_name, c1, c2, c3, c4]

    with torch.no_grad():
        with tqdm(total=len(loader), desc='Validation round', unit='batch', leave=False) as pbar:
            for i, (batch) in enumerate(loader):
                input = batch['image'].to(args.device, dtype=torch.float32)
                target = batch['label'].to(args.device, dtype=torch.long)  # shape: [N, H, W]
                if args.net == 'cats2d':
                    output, _, _ = net(input)
                elif args.net == 'mask2former':
                    outputs = net(pixel_values=input)
                    class_probs = outputs.class_queries_logits.softmax(dim=-1)[..., :-1]
                    mask_logits = outputs.masks_queries_logits
                    output = torch.einsum("bqhw,bqc->bchw", mask_logits, class_probs)
                    output = F.interpolate(output, size=target.shape[-2:], mode="bilinear", align_corners=False)
                else:
                    output = net(input)
                    if args.net == 'segformer':
                        output = output.logits
                        output = F.interpolate(output, size=(args.height, args.width), mode='bilinear',
                                               align_corners=False)

                output_prob = F.softmax(output, dim=1)

                # Get predicted class labels
                output_pred = torch.argmax(output_prob, dim=1).detach().cpu()  # shape: [N, H, W]

                if args.save_results and save_results_dir:
                    name = batch["name"][0]  # full path to original image

                    seq = os.path.basename(os.path.dirname(name))  # e.g. 2025_may07_15-28-17__needle
                    stem = os.path.splitext(os.path.basename(name))[0]  # e.g. frame_0009500
                    overlay_fname = f"{seq}_{stem}_vis.png"  # unique!

                    save_overlay_prediction(
                        original_image_path=name,
                        pred_mask_tensor=output_pred[0],
                        save_path=os.path.join(save_results_dir, overlay_fname),
                        cropped = args.cropped
                    )

                target = target.detach().cpu()

                # Compute Dice score
                mean_dice, std_dice, dice_per_class = dice_coeff(output_pred, target, num_classes=5)
                dice_records.append(mean_dice)

                # keep dice for CSV
                img_name = os.path.basename(batch["name"][0])
                class_vals = [float(d) if not torch.isnan(d) else np.nan for d in dice_per_class]

                csv_rows.append([img_name] + class_vals)

                per_image_dice.append((batch["name"][0], class_vals))

                pbar.update(1)
            pbar.close()

    # write CSV
    import csv, time
    csv_path = os.path.join(args.save_dir, f"dice_per_image_{time.strftime('%Y%m%d_%H%M%S')}.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "grasper_outer_tube", "grasper_tip", "needle", "thread"])
        writer.writerows(csv_rows)

    logging.info(f"Per-image Dice CSV written to {csv_path}")

    return dice_records, per_image_dice

def test_net_baseline(args, net1, dataset, batch_size=1):
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    logging.info(f'''Starting testing:
            Num test:        {len(dataset)}
            Batch size:      {batch_size}
            Device:          {torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')}
        ''')

    # disables dropout (no randomness)
    net1.eval()

    best_model_path = os.path.join(args.save_dir, "cp", "best_net.pth")
    print(best_model_path)
    # last_model_path = os.path.join(args.save_dir, "cp", "last_net.pth")
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # Load best model
    net1.load_state_dict(torch.load(best_model_path, map_location=device))

    dice_records, per_image_dice = validate_baseline(
        args, net1, test_loader, save_results_dir=os.path.join(args.save_dir, "results_best")
    )

    logging.info('Model, batch-wise validation Dice coeff: {:.4f}, std: {:.4f}'.format(
        np.mean(dice_records), np.std(dice_records)))

    # ---------- pick best / worst overall image (highest / lowest mean Dice) ----------
    best_idx = int(np.argmax(dice_records))
    worst_idx = int(np.argmin(dice_records))

    best_name, best_vec = per_image_dice[best_idx]
    worst_name, worst_vec = per_image_dice[worst_idx]

    CLS_IDS = [1, 2, 3, 4]

    per_class_scores = {cls: [] for cls in CLS_IDS}
    for _, dice_vec in per_image_dice:
        for i, cls in enumerate(CLS_IDS):
            per_class_scores[cls].append(dice_vec[i])

    logging.info("\nPer-class Average Dice Scores")
    for cls in CLS_IDS:
        scores = per_class_scores[cls]
        # avg_score = sum(scores) / len(scores)
        avg_score = torch.nanmean(torch.tensor(scores))

        tensor_scores = torch.tensor(scores)
        valid_scores = tensor_scores[~torch.isnan(tensor_scores)]
        std_score = torch.std(valid_scores, unbiased=False)

        logging.info(f"Class {cls} Avg Dice: {avg_score:.4f} Std Dice: {std_score:.4f}")

    # Create the output folder
    output_root = "/home/dilara/Desktop/CAO_seg-main/src/result_dice_images"
    os.makedirs(output_root, exist_ok=True)

    def build_vis_fname(img_path):
        seq = os.path.basename(os.path.dirname(img_path))
        stem = os.path.splitext(os.path.basename(img_path))[0]
        return f"{seq}_{stem}_vis.png"

    best_png = build_vis_fname(best_name)
    worst_png = build_vis_fname(worst_name)

    src_best = os.path.join(args.save_dir, "results_best", best_png)
    src_worst = os.path.join(args.save_dir, "results_best", worst_png)

    dst_best = os.path.join(output_root, best_png)
    dst_worst = os.path.join(output_root, worst_png)

    # Copy if the files exist
    if os.path.exists(src_best):
        shutil.copy(src_best, dst_best)
    if os.path.exists(src_worst):
        shutil.copy(src_worst, dst_worst)


if __name__ == '__main__':
    init_seeds(42)
    args = parser.parse_args()
    setup_logging(args, mode='test')
    logging.info(os.path.abspath(__file__))
    logging.info(args)

    assert args.json_path is not None, 'input your split file'

    dataset = get_dataset(args, mode='test')
    net = get_net(args, pretrain=True)

    test_net_baseline(args,
             net1=net,
             dataset=dataset)



