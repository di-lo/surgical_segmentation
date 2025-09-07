import os
import torch
import numpy as np
from torch.utils.data import Dataset
import glob, json
import PIL.Image as Image
import random
from transformers import SegformerFeatureExtractor
from torchvision.transforms.functional import to_pil_image

filename_types = ['png', 'jpg', 'jpeg', 'tif', 'tiff', 'PNG', 'JPG', 'JPEG', 'TIFF', 'TIFF', 'npz', 'npy']

class CAO_dataset(Dataset):
    def __init__(self, args, mode='train', transform=None):
        self.args = args
        self.mode = mode
        self.transform = transform

        if args.net == 'segformer':
            self.feature_extractor = SegformerFeatureExtractor.from_pretrained(
                "nvidia/segformer-b0-finetuned-ade-512-512"
            )
        if mode != 'test':
            json_file_path = args.json_path
            with open(json_file_path, 'r') as file:
                split = json.load(file)
            self.data = split[mode]
        else:
            if args.json_path is not None:
                with open(args.json_path, 'r') as file:
                    split = json.load(file)
                # it should be the test
                self.data = split['test']
            else:
                data = sorted(os.listdir(args.test_data_dir))
                self.data = [f for f in data
                             if f != ".DS_Store" and os.path.isfile(os.path.join(args.test_data_dir, f))]
        self.transform = transform

        #self.data = self.data[0:2]

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        if self.mode != 'test':
            image_path = self.data[idx]['image']
            label_path = self.data[idx]['label']
            img = Image.open(image_path)        # [3, Height, Width]
            if img.mode == "RGBA":
                image = np.array(img.convert("RGB"))
            else:
                image = np.array(img.convert("RGB"))
            mask = np.array(Image.open(label_path))

            mask = self.map_rgb_mask_to_class(mask)
            h, w = image.shape[:2]
        else:
            image_path = self.data[idx]['image']
            label_path = self.data[idx]['label']
            if '.npy' in image_path:
                image = np.load(image_path)
            else:
                image = Image.open(image_path)  # [3, Height, Width]
                if image.mode == "RGBA":
                    image = image.convert("RGB")
                image = np.array(image.convert("RGB"))
            mask = np.array(Image.open(label_path))
            mask = self.map_rgb_mask_to_class(mask)
            h, w = image.shape[:2]
            # mask = np.zeros((h, w), dtype=np.uint8)

        image = image[:, 420:-420, :]
        mask = mask[:, 420:-420]

        # with Albumentations
        if self.transform is not None:
            mask = np.expand_dims(mask, -1)  # Albumentations needs this shape
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            image = np.moveaxis(image, -1, 0)  # HWC → CHW
            mask = np.moveaxis(mask, -1, 0)
            return {
                'image': torch.from_numpy(image).float(),
                'label': torch.from_numpy(mask).long(),
                'name': image_path,
                'original_shape': (h, w)
            }

        # NO Albumentations
        if image.dtype != np.uint8:
            if image.max() <= 1:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        img_pil = Image.fromarray(image)
        encoded = self.feature_extractor(images=img_pil, return_tensors="pt")
        pixel_values = encoded["pixel_values"].squeeze(0)  # shape (3, H, W)

        return {
            'image': pixel_values.float(),  # key is still 'image'
            'label': torch.from_numpy(mask).long(),  # (H, W)
            'name': image_path,
            'original_shape': (h, w)
        }

    import numpy as np

    import numpy as np

    def map_rgb_mask_to_class(self, mask_rgb):
        """
        Convert an RGB mask to class indices 1–4 based on color.

        Args:
            mask_rgb (np.ndarray): RGB mask of shape (H, W, 3).

        Returns:
            np.ndarray: Class-labeled mask of shape (H, W), values in {1,2,3,4}.
        """
        # Define the mapping: (R, G, B) → class ID
        if self.args.new:
            color_to_class = {
                (135, 46, 23): 1,  # Grasper_outer_tube
                (168, 231, 243): 2,  # Grasper_tip
                (12, 88, 37): 4,  # Thread
                (174, 67, 120): 3,  # Needle
            }
        else:
            color_to_class = {
                (135, 46, 23): 1,
                (168, 231, 243): 2,
                (12, 88, 37): 3,
                (174, 67, 120): 4,
            }

        # Initialize result
        h, w, _ = mask_rgb.shape
        class_mask = np.zeros((h, w), dtype=np.uint8)

        # Loop over the mapping and assign class IDs
        for color, class_id in color_to_class.items():
            mask = np.all(mask_rgb == color, axis=-1)
            class_mask[mask] = class_id


        return class_mask


