"""
utils.py — 모든 step에서 공통으로 사용하는 함수 모음
별도로 실행할 필요 없이 각 step에서 import해서 사용
"""

import cv2, numpy as np, torch, inspect
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.feature import local_binary_pattern

# ── 경로 설정 (step마다 BASE_DIR을 import하거나 여기서 통일) ──────────────────
BASE_DIR = "/content/drive/MyDrive/forensic_project"

# ── 1. Letterbox Padding Resize ───────────────────────────────────────────────
def letterbox_resize(image, target=256, mask=None):
    """
    이미지 비율을 유지하면서 target×target 크기로 맞춤.
    빈 공간은 검정(0)으로 패딩.
    mask가 주어지면 동일한 변환을 적용 (nearest neighbor).
    """
    h, w = image.shape[:2]
    scale = target / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_h = target - new_h
    pad_w = target - new_w
    top, left = pad_h // 2, pad_w // 2
    bottom = pad_h - top
    right  = pad_w - left

    padded_img = cv2.copyMakeBorder(
        resized_img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=0
    )

    if mask is not None:
        resized_msk = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        padded_msk  = cv2.copyMakeBorder(
            resized_msk, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=0
        )
        return padded_img, padded_msk

    return padded_img


# ── 2. Dataset 클래스 ─────────────────────────────────────────────────────────
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _make_gauss_noise():
    """Albumentations 1.x/2.x 모두에서 동작하도록 Gaussian noise transform 생성."""
    params = inspect.signature(A.GaussNoise).parameters
    if "var_limit" in params:
        return A.GaussNoise(var_limit=(10.0, 50.0), p=0.3)
    return A.GaussNoise(std_range=(0.03, 0.12), mean_range=(0.0, 0.0), p=0.3)


def _make_image_compression():
    """Albumentations 1.x/2.x 모두에서 동작하도록 JPEG compression transform 생성."""
    params = inspect.signature(A.ImageCompression).parameters
    if "quality_lower" in params:
        return A.ImageCompression(quality_lower=50, quality_upper=95, p=0.4)
    return A.ImageCompression(quality_range=(50, 95), p=0.4)


def get_augmentation(mode="train"):
    """
    mode='train': 증강 포함
    mode='val'  : 증강 없이 정규화만
    """
    if mode == "train":
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.4, border_mode=cv2.BORDER_CONSTANT),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            _make_gauss_noise(),
            _make_image_compression(),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ], additional_targets={"mask": "mask"})
    else:
        return A.Compose([
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ], additional_targets={"mask": "mask"})


class ForensicDataset(Dataset):
    """
    이미지 + mask를 읽어서 letterbox resize 후 반환하는 Dataset.
    label: 0=real, 1=composite
    """
    def __init__(self, df, mode="train", target_size=256):
        self.df          = df.reset_index(drop=True)
        self.target_size = target_size
        self.transform   = get_augmentation(mode)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 이미지 읽기
        img = cv2.imread(row["image_path"])
        if img is None:
            raise FileNotFoundError(f"이미지 없음: {row['image_path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # mask 읽기 (흑백)
        msk = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
        if msk is None:
            msk = np.zeros(img.shape[:2], dtype=np.uint8)
        msk = (msk > 127).astype(np.uint8)   # binary (0 or 1)

        # Letterbox resize
        img, msk = letterbox_resize(img, self.target_size, msk)

        # 증강 + 정규화 + tensor 변환
        aug   = self.transform(image=img, mask=msk.astype(np.float32))
        image = aug["image"]          # float32 tensor (3, H, W)
        mask  = aug["mask"].unsqueeze(0).float()  # (1, H, W)

        label = torch.tensor(row["label"], dtype=torch.long)
        return image, mask, label


# ── 3. Loss 함수 (BCE + Dice) ─────────────────────────────────────────────────
import torch.nn as nn

class BCEDiceLoss(nn.Module):
    """
    L = BCE(pos_weight) + λ × Dice
    - BCE: real image의 all-zero mask에서 항상 0만 예측하는 편향 방지
    - Dice: 작은 조작 영역을 놓치지 않도록 예측·정답 겹침 비율 직접 최적화
    """
    def __init__(self, lam=1.0, pos_weight=2.0):
        super().__init__()
        self.lam = lam
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

    def dice_loss(self, pred, target, smooth=1e-6):
        pred   = torch.sigmoid(pred)
        inter  = (pred * target).sum(dim=(2, 3))
        union  = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice   = (2 * inter + smooth) / (union + smooth)
        return 1 - dice.mean()

    def forward(self, pred, target):
        self.bce.pos_weight = self.bce.pos_weight.to(pred.device)
        return self.bce(pred, target) + self.lam * self.dice_loss(pred, target)


# ── 4. Stage 1 평가지표 ────────────────────────────────────────────────────────
def compute_stage1_metrics(pred_np, gt_np, threshold=0.5):
    """
    pred_np: sigmoid heatmap (H×W, float 0~1)
    gt_np  : ground-truth mask (H×W, binary 0/1)
    반환: dict {iou, dice, precision, recall, fp_area_ratio}
    """
    pred_bin = (pred_np > threshold).astype(np.uint8)
    gt_bin   = (gt_np  > 0.5).astype(np.uint8)

    inter = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum()
    pred_sum = pred_bin.sum()
    gt_sum = gt_bin.sum()

    # GT가 비어 있는 real 이미지에서는 pred도 비어 있으면 완전 정답으로 처리한다.
    # 이렇게 하지 않으면 real 이미지의 perfect all-zero 예측 Dice가 0으로 계산된다.
    if gt_sum == 0:
        fp_area_ratio = pred_bin.mean()
        if pred_sum == 0:
            iou = dice = precision = recall = 1.0
        else:
            iou = dice = precision = 0.0
            recall = 1.0
    else:
        iou       = inter / (union + 1e-6)
        dice      = 2 * inter / (pred_sum + gt_sum + 1e-6)
        precision = inter / (pred_sum + 1e-6)
        recall    = inter / (gt_sum + 1e-6)
        fp_area_ratio = np.nan

    return {
        "iou"           : float(iou),
        "dice"          : float(dice),
        "precision"     : float(precision),
        "recall"        : float(recall),
        "fp_area_ratio" : float(fp_area_ratio) if not np.isnan(fp_area_ratio) else None,
    }


# ── 5. Feature Extraction ─────────────────────────────────────────────────────
def extract_features(image_rgb, heatmap, threshold=0.5, top_k=5, min_area=100):
    """
    image_rgb : (H, W, 3) uint8 RGB
    heatmap   : (H, W) float32 0~1 (Stage 1 sigmoid 출력)
    반환: dict with ~16 features

    핵심 흐름:
    heatmap → threshold → binary mask → connected component
    → min_area 필터 → top-k 의심 영역 선택
    → 밝기·색상·경계·texture·global feature 계산
    """
    feats = {}

    # ─ heatmap confidence features (항상 계산)
    feats["max_suspicious_score"]  = float(heatmap.max())
    feats["mean_suspicious_score"] = float(heatmap.mean())

    binary = (heatmap > threshold).astype(np.uint8)
    feats["suspicious_area_ratio"] = float(binary.mean())

    # ─ connected component + min_area 필터
    num_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(binary)

    valid_regions = []
    for lid in range(1, num_labels):   # 0은 배경
        area = stats[lid, cv2.CC_STAT_AREA]
        if area >= min_area:
            region_mask = (label_map == lid)
            valid_regions.append({
                "lid"  : lid,
                "area" : area,
                "score": float(heatmap[region_mask].mean()),
                "bbox" : stats[lid, :4].tolist(),   # x, y, w, h
            })

    valid_regions.sort(key=lambda x: x["score"], reverse=True)
    top_regions = valid_regions[:top_k]

    if top_regions:
        feats["top_k_score_mean"] = float(np.mean([r["score"] for r in top_regions]))

        # top-k 영역을 하나의 mask로 합치기
        roi_mask = np.zeros(binary.shape, dtype=np.uint8)
        for r in top_regions:
            roi_mask[label_map == r["lid"]] = 1

        # 주변(surrounding) 영역: roi를 팽창 후 roi 제거
        kernel = np.ones((25, 25), np.uint8)
        dilated = cv2.dilate(roi_mask, kernel)
        surr_mask = (dilated - roi_mask).astype(bool)
        roi_bool  = roi_mask.astype(bool)

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        if roi_bool.sum() > 0 and surr_mask.sum() > 0:
            # ─ 밝기 features
            roi_gray  = gray[roi_bool].astype(float)
            surr_gray = gray[surr_mask].astype(float)
            feats["brightness_diff"]    = float(abs(roi_gray.mean() - surr_gray.mean()))
            feats["roi_brightness_std"] = float(roi_gray.std())

            # ─ 색상 features
            roi_rgb  = image_rgb[roi_bool].astype(float)
            surr_rgb = image_rgb[surr_mask].astype(float)
            feats["rgb_mean_diff"]  = float(np.abs(roi_rgb.mean(0) - surr_rgb.mean(0)).mean())
            feats["color_distance"] = float(np.linalg.norm(roi_rgb.mean(0) - surr_rgb.mean(0)))

            hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV).astype(float)
            feats["hsv_saturation_diff"] = float(abs(hsv[:,:,1][roi_bool].mean() - hsv[:,:,1][surr_mask].mean()))
            feats["hsv_value_diff"]      = float(abs(hsv[:,:,2][roi_bool].mean() - hsv[:,:,2][surr_mask].mean()))

            # ─ 경계 features (roi 경계 픽셀에서 계산)
            eroded   = cv2.erode(roi_mask, np.ones((3,3),np.uint8))
            boundary = (roi_mask - eroded).astype(bool)

            edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
            feats["boundary_edge_strength"] = float(edges[boundary].mean()) if boundary.sum() > 0 else 0.0

            gx = cv2.Sobel(gray.astype(np.uint8), cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray.astype(np.uint8), cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(gx**2 + gy**2)
            feats["boundary_gradient_mean"] = float(grad_mag[boundary].mean()) if boundary.sum() > 0 else 0.0

            lap = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F)
            feats["laplacian_variance"] = float(lap[roi_bool].var())

            # ─ LBP texture feature
            lbp = local_binary_pattern(gray.astype(np.uint8), P=8, R=1, method="uniform")
            bins = 10
            h_roi,  _ = np.histogram(lbp[roi_bool],  bins=bins, range=(0, bins), density=True)
            h_surr, _ = np.histogram(lbp[surr_mask], bins=bins, range=(0, bins), density=True)
            feats["lbp_hist_diff"] = float(np.abs(h_roi - h_surr).sum())

        else:
            # roi 또는 surrounding이 비어있으면 0으로
            for k in ["brightness_diff","roi_brightness_std","rgb_mean_diff","color_distance",
                      "hsv_saturation_diff","hsv_value_diff","boundary_edge_strength",
                      "boundary_gradient_mean","laplacian_variance","lbp_hist_diff"]:
                feats[k] = 0.0
    else:
        # 최소 면적 통과 영역 없음 → 모두 0
        feats["top_k_score_mean"] = 0.0
        for k in ["brightness_diff","roi_brightness_std","rgb_mean_diff","color_distance",
                  "hsv_saturation_diff","hsv_value_diff","boundary_edge_strength",
                  "boundary_gradient_mean","laplacian_variance","lbp_hist_diff"]:
            feats[k] = 0.0

    # ─ global features (항상 계산 - Stage 1 오류 시 보완)
    gray_full = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    feats["global_brightness_mean"] = float(gray_full.mean())
    feats["global_edge_mean"]       = float(cv2.Canny(gray_full, 50, 150).mean())

    # LightGBM 입력 안정성을 위해 NaN/Inf 제거
    for key, value in list(feats.items()):
        feats[key] = float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))

    return feats


# feature 컬럼 정의 (버전별로 선택해서 사용)
FEATURE_GROUPS = {
    "confidence": ["max_suspicious_score","mean_suspicious_score",
                   "suspicious_area_ratio","top_k_score_mean"],
    "brightness": ["brightness_diff","roi_brightness_std"],
    "color"     : ["rgb_mean_diff","color_distance","hsv_saturation_diff","hsv_value_diff"],
    "edge"      : ["boundary_edge_strength","boundary_gradient_mean",
                   "laplacian_variance","lbp_hist_diff"],
    "global"    : ["global_brightness_mean","global_edge_mean"],
}

FEATURE_VERSIONS = {
    "Baseline"  : FEATURE_GROUPS["global"],
    "Version1"  : FEATURE_GROUPS["confidence"] + FEATURE_GROUPS["brightness"],
    "Version2"  : FEATURE_GROUPS["confidence"] + FEATURE_GROUPS["color"],
    "Version3"  : FEATURE_GROUPS["confidence"] + FEATURE_GROUPS["edge"],
    "Version4"  : (FEATURE_GROUPS["confidence"] + FEATURE_GROUPS["brightness"] +
                   FEATURE_GROUPS["color"] + FEATURE_GROUPS["edge"] + FEATURE_GROUPS["global"]),
}

ALL_FEATURE_COLS = (FEATURE_GROUPS["confidence"] + FEATURE_GROUPS["brightness"] +
                    FEATURE_GROUPS["color"] + FEATURE_GROUPS["edge"] + FEATURE_GROUPS["global"])
