from __future__ import annotations

import cv2
import json
import random
import argparse

from typing import List
from pathlib import Path
from ultralytics import YOLO
from packaging import version

import numpy as np
import albumentations as A

try:
    import albumentations
    _ALBU_VER = version.parse(albumentations.__version__)
except Exception:
    _ALBU_VER = version.parse("1.0.0")


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------
# Albumentations pipeline for training batches via callbacks
# ---------------------------

def _coarse_dropout(imgsz: int, occl_prob: float):
    kwargs_common = dict(
        max_holes=10,
        max_height=int(0.12*imgsz),
        max_width=int(0.12*imgsz),
        min_holes=1,
        min_height=8,
        min_width=8,
        p=occl_prob
    )
    # Albu v2+ uses 'fill'; v1 uses 'fill_value'
    try:
        if _ALBU_VER >= version.parse("2.0.0"):
            return A.CoarseDropout(fill=114, **kwargs_common)
        else:
            return A.CoarseDropout(fill_value=114, **kwargs_common)
    except TypeError:
        # last-resort fallback
        return A.CoarseDropout(fill_value=114, **kwargs_common)

def build_albu_pipeline(imgsz=640, rain_snow_prob=0.5, occl_prob=0.5):
    return A.Compose([
        A.LongestMaxSize(imgsz + 64, p=1.0),
        A.PadIfNeeded(imgsz, imgsz, border_mode=cv2.BORDER_CONSTANT, value=(114,114,114), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.15, rotate_limit=5,
                           border_mode=cv2.BORDER_CONSTANT, value=(114,114,114), p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(0.3, 0.3, p=0.8),
            A.CLAHE(tile_grid_size=(8,8), clip_limit=2.0, p=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
        ], p=0.9),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 9), p=0.5),
            A.MotionBlur(blur_limit=11, p=0.5),
            A.GlassBlur(sigma=0.7, max_delta=2, p=0.3)
        ], p=0.5),
        A.ImageCompression(quality_lower=45, quality_upper=95, p=0.4),
        A.OneOf([
            A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, p=0.3),
            A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.5, alpha_coef=0.08, p=0.3),
            A.RandomSunFlare(flare_roi=(0,0,1,0.5), angle_lower=0.0, src_radius=80, p=0.3),
            A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1,
                         blur_value=3, brightness_coefficient=0.95, p=1.0),
            A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=1.0, p=1.0),
        ], p=rain_snow_prob),
        _coarse_dropout(imgsz, occl_prob),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.2))


def make_callback(albu):
    """Ultralytics callback to apply Albumentations on preprocessed batches."""
    import torch

    def _to_numpy_img(t):
        # t: CxHxW float32 [0,1] -> HxWxC uint8
        arr = (t.permute(1,2,0).cpu().numpy() * 255.0).clip(0,255).astype(np.uint8)
        return arr

    def _to_tensor_img(arr):
        # arr: HxWxC uint8 -> CxHxW float32 [0,1]
        return torch.from_numpy(arr.astype(np.float32) / 255.0).permute(2,0,1)

    def _ensure_list(x):
        # some YOLO builds pack as list[Tensor], others as Tensor[list]
        return x if isinstance(x, (list, tuple)) else list(x)

    def on_preprocess_batch(trainer):
        batch = trainer.batch
        imgs = batch['img']  # (B, C, H, W) float32 0-1
        bboxes = _ensure_list(batch.get('bboxes', []))
        classes = _ensure_list(batch.get('cls', []))
        B = imgs.shape[0]
        new_imgs, new_bboxes, new_classes = [], [], []
        for i in range(B):
            img_np = _to_numpy_img(imgs[i])
            # safely get boxes/classes for sample i
            boxes_i = bboxes[i].cpu().numpy() if i < len(bboxes) and bboxes[i] is not None else np.zeros((0,4), dtype=np.float32)
            cls_i = classes[i].cpu().numpy().reshape(-1) if i < len(classes) and classes[i] is not None else np.zeros((0,), dtype=np.float32)
            if boxes_i.ndim == 1 and boxes_i.size == 0:
                boxes_i = np.zeros((0,4), dtype=np.float32)
            if cls_i.ndim == 0:
                cls_i = np.array([cls_i], dtype=np.float32)
            if len(cls_i) != len(boxes_i):
                cls_i = cls_i[:len(boxes_i)]
            try:
                out = albu(image=img_np, bboxes=boxes_i.tolist(), class_labels=cls_i.tolist())
                img_aug = out['image']
                bbs_aug = np.array(out.get('bboxes', []), dtype=np.float32) if out.get('bboxes', []) else np.zeros((0,4), dtype=np.float32)
                cls_aug = np.array(out.get('class_labels', []), dtype=np.float32).reshape(-1,1) if out.get('class_labels', []) else np.zeros((0,1), dtype=np.float32)
            except Exception:
                # Fallback: keep original in case augmentation fails for a sample
                img_aug = img_np
                bbs_aug = boxes_i
                cls_aug = np.array(cls_i, dtype=np.float32).reshape(-1,1)
            new_imgs.append(_to_tensor_img(img_aug))
            new_bboxes.append(torch.from_numpy(bbs_aug))
            new_classes.append(torch.from_numpy(cls_aug))
        batch['img'] = torch.stack(new_imgs, dim=0).to(imgs.device)
        batch['bboxes'] = new_bboxes
        batch['cls'] = new_classes

    return {
        'on_preprocess_batch': on_preprocess_batch
    }


# ---------------------------
# Train wrapper (uses existing data.yaml)
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_yaml', type=Path, default=Path('./datasets/data.yaml'),
                    help='Path to YOLO data.yaml with train/val[/test] and class names.')
    ap.add_argument('--name', type=str, default='car_damage')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=120)
    ap.add_argument('--warmup_epochs', type=int, default=3)
    ap.add_argument('--patience', type=int, default=30)
    ap.add_argument('--device', type=str, default='0')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--weights', type=str, default='yolov8s.pt')
    ap.add_argument('--resume_if_exists', action='store_true', default=True)
    ap.add_argument('--rain_snow_prob', type=float, default=0.6)
    ap.add_argument('--occl_prob', type=float, default=0.5)
    ap.add_argument('--project', type=str, default='runs/detect')
    args = ap.parse_args()

    random.seed(SEED); np.random.seed(SEED)

    # 1) Verify data.yaml exists and looks sane
    data_yaml = args.data_yaml
    assert data_yaml.exists(), f"data.yaml not found at: {data_yaml}"

    # 2) Albumentations + callbacks
    albu = build_albu_pipeline(imgsz=args.imgsz, rain_snow_prob=args.rain_snow_prob, occl_prob=args.occl_prob)
    callbacks = make_callback(albu)

    # 3) Train
    model = YOLO(args.weights)  # transfer learning from COCO

    # Register callbacks with the model (supported API)
    for name, fn in callbacks.items():
        model.add_callback(name, fn)

    # Training args (pass as kwargs)
    train_kwargs = dict(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        lr0=0.002,           # slightly higher initial LR for s-model
        lrf=0.1,             # one-cycle final LR fraction
        momentum=0.937,
        weight_decay=0.0005,
        degrees=0.0,         # disable built-ins we emulate via Albumentations
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,          # keep mosaic off with heavy albu
        mixup=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        project=args.project,
        name=args.name,
        cache='ram',         # speed on A100 w/ enough RAM
        close_mosaic=10,
    )

    resume = False
    exp_dir = Path(args.project) / args.name
    if args.resume_if_exists and (exp_dir / 'weights' / 'last.pt').exists():
        resume = True

    model.train(resume=resume, **train_kwargs)

    # 4) Validate & export (uses same data.yaml)
    metrics = model.val(data=str(data_yaml), imgsz=args.imgsz, device=args.device, plots=True)
    print(metrics)

    # Optional export to ONNX
    try:
        model.export(format='onnx', dynamic=True)
    except Exception as e:
        print('Export warning:', e)


if __name__ == '__main__':
    main()
