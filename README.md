# Surgical Instrument Segmentation for Central Airway Suturing

> A clean, reproducible codebase accompanying our benchmark study on segmenting instruments in **Central Airway Obstruction (CAO) suturing**.

<p align="center">
  <img src="assets/teaser.gif" alt="Qualitative segmentation preview" width="900"/>
</p>

---

## TL;DR

* **Train**: `python src/train.py --json_path <PATH/TO/split.json> --name <checkpoints/model_name> --mode train --batch_size <2|8> --net <unet|deeplabplus|cats2d|dscnet|segformer|mask2former|swinunet|cats> --height 512 --width 512`
* **Test**: `python src/test.py --json_path <PATH/TO/split.json> --save_results -- net <unet|deeplabplus|cats2d|dscnet|segformer|mask2former|swinunet|cats> --mode test --height 512 --width 512`
* **SAM2 (optional)**: Download `sam2_hiera_large.pt` and place it under `src/sam2_checkpoints/`.

---

## Motivation

Segmentation under **domain shift** (lighting/background changes, motion blur, small reflective targets like needles, thin deformable thread) is hard. This repo implements and compares CNN-, Transformer-, and Hybrid-based approaches, with/without **pretrained encoders**, on a real CAO suturing dataset.

---

## What’s inside

### Architectures

![Deep Learning Architectures](dl_architectures.png)

* **CNN**: U‑Net (ResNet‑34 encoder), DeepLabv3+ (ResNet‑34), **DSCNet** (dynamic snake convolution; no pretraining)
* **Transformer**: **SegFormer** (MiT), **Mask2Former** (Swin), **SAM2** encoder variants (FT/PEFT)
* **Hybrid**: **CATS / CATS2d** — CNN+Transformer hybrid; ViT/SAM2‑Hiera encoder with CNN decoder and U‑Net‑style skips

> All models share a unified training/testing interface.

### Dataset (expected layout)

* 18 HD endoscopic videos (1920×1080). We use 5 videos (656 frames) for **train**, 2 (132) **in‑domain test**, 11 (783) **out‑of‑domain test**.
* Images are **center‑cropped** to 1080×1080 and **resized** to 512×512 for training/eval.
* 4 classes: **outer tube**, **inner tube**, **needle**, **thread**.

> Use your own data or request access to the study dataset.

```
data/
├── split/
│ ├── make_train_val_split.py # 80/20 split -> splits/train.json, splits/val.json
│ └── split.json
└── test/
  ├── make_test_json.py
  └── split.json
```

---

## Installation

```bash
# clone
git clone https://github.com/MedICL-VU/needle_seg.git
cd needle_seg

# environment (Python 3.9–3.11 supported)
conda create -n needle_seg python=3.10 -y
conda activate needle_seg

# deps
pip install -r requirements.txt
```

**Repo structure**

```
needle_seg/
├── src/
│   ├── config_setup.py
│   ├── config_args.py
│   ├── utils.py
│   ├── CAO_dataset.py
│   ├── DSCNet.py
│   ├── cats.py
│   ├── CATS2d.py
│   ├── train.py
│   ├── test.py
│   └── sam2_checkpoints/
├── assets/
│   └── teaser.gif
├── requirements.txt
└── README.md
```

---

**Notes**

* **Loss**: Cross‑Entropy + Dice.
* **Optimizer**: AdamW.
* **Epochs**: 100.
* **Batch size**: 8 typically; use **2** for heavy models (e.g., SAM2 FT, DSCNet) if VRAM limited.
* **Device**: single‑GPU training is supported. A4500/A6000‑class GPUs recommended.

---

## Reproducing paper results

* For each architecture, run **with** and **without** pretrained encoders.
* Report **mean Dice** over in‑domain and out‑of‑domain sets.
* Expect transformer‑based models + **pretraining** to be stronger under domain shift; **Mask2Former** typically performs best on OOD across classes; **CATS** is strong on grasper tubes; **DSCNet** is robust on curved, thin structures (needle/thread).

> For exact numbers, qualitative figures, and epoch‑wise analyses, see the paper and replicate the training schedule above.

---

## Implementation details

* **CATS / CATS2d**: Hybrid encoder (CNN + Transformer). We wrap a **SAM2 Hiera‑Large** image encoder and use a CNN decoder with **U‑Net‑style skip connections** and lightweight **RFB** blocks for multi‑scale fusion. Encoder can be **frozen** (PEFT‑style adapters) or **fully fine‑tuned**.
* **SegFormer / Mask2Former**: HuggingFace/torchvision implementations with MiT/Swin backbones; we expose minimal flags to toggle pretraining and image size.
* **DSCNet**: Dynamic snake convolution for tubular/curvilinear structures; trained from scratch.
* **UNet / DeepLabv3+**: ResNet‑34 encoders with ImageNet weights via torchvision; standard decoders.

---

## Results preview

* **Pretraining helps** across all models.
* **Transformers** tend to generalize better **out‑of‑domain**.
* **Mask2Former** excels on **thread** delineation and provides the best OOD averages in our study.
* **CATS** is particularly reliable on **outer/inner grasper tubes**; **DSCNet** does well on **needle/thread** without pretraining.

> See the paper for full tables/plots and qualitative examples.

---

## Citation

If you use this repo, please cite the accompanying paper:

comming soon

---

## License

The model is licensed under the Apache 2.0 license

---

## Contact

Please send an email to dilara.isik@vanderbilt.edu or hao.li.1@vanderbilt.edu

---

## Changelog

* 2025‑09‑07: Initial public README draft.
