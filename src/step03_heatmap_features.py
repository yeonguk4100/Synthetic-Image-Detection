"""
Step 03 — Heatmap 생성 + Feature 추출

흐름:
  1. 최종 Stage 1 체크포인트로 train/validation/test 전체 heatmap 생성 → 저장
  2. 각 heatmap에서 feature vector 추출
  3. threshold / top-k 조합별로 feature CSV 저장
"""

import os, sys, json
import numpy as np, pandas as pd, torch, cv2
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, "/content/drive/MyDrive/forensic_project")
from utils import (ForensicDataset, letterbox_resize, extract_features,
                   ALL_FEATURE_COLS, FEATURE_VERSIONS)

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR      = "/content/drive/MyDrive/forensic_project"
CSV_PATH      = f"{BASE_DIR}/data/dataset.csv"
LOG_PATH      = f"{BASE_DIR}/results/stage1_training_log.json"
HEATMAP_DIR   = f"{BASE_DIR}/heatmaps"
FEATURE_DIR   = f"{BASE_DIR}/features"
os.makedirs(HEATMAP_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)

# ── 탐색할 후처리 조합 ────────────────────────────────────────────────────────
THRESHOLDS = [0.4, 0.5, 0.6]
TOP_KS     = [3, 5]
MIN_AREA   = 100    # connected component 최소 면적 (px²)
TARGET_SIZE = 256
BATCH_SIZE  = 32
NUM_WORKERS = 2


def build_model():
    return smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights=None,   # 가중치는 체크포인트에서 로드
        in_channels=3,
        classes=1,
        activation=None,
    )


# ── 1. 최적 체크포인트 경로 로드 ───────────────────────────────────────────────
with open(LOG_PATH) as f:
    log = json.load(f)
BEST_CKPT = log["best_ckpt_path"]
print(f"✅ Stage 1 체크포인트: {BEST_CKPT}")


# ── 2. 전체 데이터 heatmap 생성 ───────────────────────────────────────────────
print("\n🔥 Heatmap 생성 중 (train + validation + test)...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = build_model().to(device)
model.load_state_dict(torch.load(BEST_CKPT, map_location=device))
model.eval()

df = pd.read_csv(CSV_PATH)

# Dataset (증강 없이 전처리만)
full_ds = ForensicDataset(df, mode="val", target_size=TARGET_SIZE)
full_dl = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=False,
                     num_workers=NUM_WORKERS, pin_memory=True)

heatmap_paths = []   # 행 순서대로 heatmap 경로 저장

with torch.no_grad():
    idx = 0
    for images, masks, labels in tqdm(full_dl, desc="heatmap 생성"):
        images = images.to(device)
        preds  = torch.sigmoid(model(images)).cpu().numpy()  # (B, 1, H, W)

        for pred in preds:
            row        = df.iloc[idx]
            stem       = Path(row["image_path"]).stem
            split      = row["split"]
            label      = int(row["label"])
            save_path  = f"{HEATMAP_DIR}/{idx:06d}_{split}_y{label}_{stem}.png"

            # float32/float16 npy로 저장하면 30,000장 기준 Drive 용량과 I/O가 너무 커진다.
            # 0~255 uint8 PNG로 저장하고, 로드할 때 0~1 float으로 복원한다.
            hm_u8 = np.clip(pred[0] * 255.0, 0, 255).astype(np.uint8)
            cv2.imwrite(save_path, hm_u8)
            heatmap_paths.append(save_path)
            idx += 1

if len(heatmap_paths) != len(df):
    raise RuntimeError(f"heatmap 개수 불일치: {len(heatmap_paths)} vs {len(df)}")

df["heatmap_path"] = heatmap_paths
df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")   # heatmap_path 컬럼 추가해서 덮어씀
print(f"✅ Heatmap {len(df)}개 저장 완료 → {HEATMAP_DIR}/")


# ── 3. Feature 추출 (threshold × top-k 조합별) ────────────────────────────────
print("\n⚙  Feature 추출 중...")

def load_image_rgb(img_path, target_size=TARGET_SIZE):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"이미지 없음: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = letterbox_resize(img, target_size)
    return img


def load_heatmap(path):
    path = str(path)
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)
    hm = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if hm is None:
        raise FileNotFoundError(f"heatmap 없음: {path}")
    return hm.astype(np.float32) / 255.0

for thr in THRESHOLDS:
    for k in TOP_KS:
        combo_name = f"thr{int(thr*10)}_k{k}"
        save_path  = f"{FEATURE_DIR}/features_{combo_name}.csv"

        if os.path.exists(save_path):
            print(f"  이미 존재, skip: {save_path}")
            continue

        rows = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  [{combo_name}]"):
            image_rgb  = load_image_rgb(row["image_path"])
            heatmap    = load_heatmap(row["heatmap_path"])

            feats = extract_features(
                image_rgb, heatmap,
                threshold=thr, top_k=k, min_area=MIN_AREA
            )
            feats["image_path"]   = row["image_path"]
            feats["mask_path"]    = row["mask_path"]
            feats["heatmap_path"] = row["heatmap_path"]
            feats["label"]        = row["label"]
            feats["split"]        = row["split"]
            feats["dataset_name"] = row["dataset_name"] if "dataset_name" in row else ""
            feats["manipulation_type"] = row["manipulation_type"] if "manipulation_type" in row else ""
            feats["group_id"]     = row["group_id"]
            rows.append(feats)

        feat_df = pd.DataFrame(rows)
        feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        feat_df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"  ✅ 저장: {save_path}  ({len(feat_df)}행)")

print("\n✅ Feature 추출 완료")
print(f"   생성된 조합 수: {len(THRESHOLDS) * len(TOP_KS)}개")
print(f"   저장 위치: {FEATURE_DIR}/")
