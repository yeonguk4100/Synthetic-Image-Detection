"""
Step 02 — Stage 1 U-Net 학습 (Repeated Hold-out, 3회)

흐름:
  train 데이터를 group_id 기준으로 3번 무작위 분할 (내부 train 80% / 내부 val 20%)
  → 각 분할로 U-Net 학습 (max 50 epoch, early stopping patience=10)
  → 3개 best checkpoint 중 내부 val Dice 최고인 1개를 최종 Stage 1 모델로 선택
  → 최종 모델을 외부 validation set으로 최종 평가
"""

import os, sys, json, random
import numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, "/content/drive/MyDrive/forensic_project")
from utils import ForensicDataset, BCEDiceLoss, compute_stage1_metrics, letterbox_resize
import cv2

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR    = "/content/drive/MyDrive/forensic_project"
CSV_PATH    = f"{BASE_DIR}/data/dataset.csv"
CKPT_DIR    = f"{BASE_DIR}/checkpoints"
RESULT_PATH = f"{BASE_DIR}/results/stage1_training_log.json"
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(f"{BASE_DIR}/results", exist_ok=True)
torch.backends.cudnn.benchmark = True

# ── 학습 하이퍼파라미터 ────────────────────────────────────────────────────────
CFG = {
    "target_size"   : 256,
    "batch_size"    : 16,
    "max_epochs"    : 50,
    "patience"      : 10,       # early stopping: val Dice 개선 없으면 멈춤
    "warmup_epochs" : 5,        # encoder 동결 기간
    "lam_dice"      : 1.0,      # BCE + lam × Dice
    "pos_weight"    : 2.0,      # BCE positive weight (조작 픽셀 가중치)
    "n_repeats"     : 3,        # Repeated Hold-out 반복 횟수
    "seeds"         : [42, 123, 456],
    "inner_val_ratio": 0.20,    # train 내부 val 비율
    "num_workers"   : 2,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model():
    """U-Net + EfficientNet-B4 (ImageNet 사전학습)"""
    model = smp.Unet(
        encoder_name    = "efficientnet-b4",
        encoder_weights = "imagenet",
        in_channels     = 3,
        classes         = 1,
        activation      = None,   # sigmoid는 loss에서 내부적으로 처리
    )
    return model


def get_optimizers(model, warmup=True):
    """
    warmup=True : encoder 동결, decoder만 lr=1e-3으로 학습 (1~5 epoch)
    warmup=False: encoder lr=1e-5, decoder lr=1e-4 (6 epoch 이후)
    """
    encoder_params = list(model.encoder.parameters())
    decoder_params = (list(model.decoder.parameters()) +
                      list(model.segmentation_head.parameters()))

    if warmup:
        for p in encoder_params:
            p.requires_grad = False
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3, weight_decay=1e-4
        )
    else:
        for p in encoder_params:
            p.requires_grad = True
        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": 1e-5},
            {"params": decoder_params, "lr": 1e-4},
        ], weight_decay=1e-4)

    return optimizer


def train_one_epoch(model, loader, optimizer, criterion, device, warmup):
    model.train()
    if warmup:
        model.encoder.eval()   # encoder BN을 eval 상태로 유지 (동결)

    total_loss = 0.0
    for images, masks, _ in loader:
        images = images.to(device)
        masks  = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_dice   = []

    for images, masks, _ in loader:
        images = images.to(device)
        masks  = masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        total_loss += loss.item()

        preds = torch.sigmoid(outputs).cpu().numpy()
        gts   = masks.cpu().numpy()
        for pred, gt in zip(preds, gts):
            # localization Dice는 조작 영역이 있는 composite mask 기준으로만 평균
            # real(all-zero)은 별도 Real FP Area Ratio로 평가
            if gt[0].sum() > 0:
                m = compute_stage1_metrics(pred[0], gt[0])
                all_dice.append(m["dice"])

    mean_dice = float(np.mean(all_dice)) if all_dice else 0.0
    return total_loss / len(loader), mean_dice


def train_single_run(inner_train_df, inner_val_df, run_id, seed, device):
    """
    단일 Hold-out 학습 1회.
    best checkpoint 경로와 best val Dice를 반환.
    """
    set_seed(seed)
    print(f"\n  학습 데이터: {len(inner_train_df)}장  |  내부 val: {len(inner_val_df)}장")

    train_ds = ForensicDataset(inner_train_df, mode="train",  target_size=CFG["target_size"])
    val_ds   = ForensicDataset(inner_val_df,   mode="val",    target_size=CFG["target_size"])
    train_dl = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,
                          num_workers=CFG["num_workers"], pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False,
                          num_workers=CFG["num_workers"], pin_memory=True)

    model     = build_model().to(device)
    criterion = BCEDiceLoss(lam=CFG["lam_dice"], pos_weight=CFG["pos_weight"]).to(device)
    ckpt_path = f"{CKPT_DIR}/run{run_id}_best.pth"

    best_dice   = -1.0
    no_improve  = 0
    epoch_log   = []

    for epoch in range(1, CFG["max_epochs"] + 1):
        warmup = (epoch <= CFG["warmup_epochs"])

        # epoch 전환 시 optimizer 교체
        if epoch == 1:
            optimizer = get_optimizers(model, warmup=True)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=CFG["max_epochs"], eta_min=1e-6
            )
        elif epoch == CFG["warmup_epochs"] + 1:
            print(f"  → epoch {epoch}: encoder unfreeze, differential lr 적용")
            optimizer = get_optimizers(model, warmup=False)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=CFG["max_epochs"] - CFG["warmup_epochs"], eta_min=1e-6
            )

        train_loss = train_one_epoch(model, train_dl, optimizer, criterion, device, warmup)
        val_loss, val_dice = validate(model, val_dl, criterion, device)
        scheduler.step()

        print(f"  Epoch {epoch:3d}/{CFG['max_epochs']} | "
              f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_dice={val_dice:.4f}"
              + (" ← best" if val_dice > best_dice else ""))

        epoch_log.append({"epoch": epoch, "train_loss": train_loss,
                          "val_loss": val_loss, "val_dice": val_dice})

        if val_dice > best_dice:
            best_dice = val_dice
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= CFG["patience"]:
                print(f"  ⛔ Early stopping at epoch {epoch} (patience={CFG['patience']})")
                break

    print(f"  ✅ Run {run_id} 완료 | best val Dice = {best_dice:.4f}")
    return best_dice, ckpt_path, epoch_log


# ── 메인 실행 ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥  device: {device}")

    df       = pd.read_csv(CSV_PATH)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "validation"].reset_index(drop=True)
    print(f"📊 train={len(train_df)}장  |  validation={len(val_df)}장")
    print(pd.crosstab(df["split"], df["label"]))
    missing_img = (~df["image_path"].map(os.path.exists)).sum()
    missing_msk = (~df["mask_path"].map(os.path.exists)).sum()
    if missing_img or missing_msk:
        raise FileNotFoundError(f"경로 오류: missing images={missing_img}, missing masks={missing_msk}")

    run_results = []
    best_overall_dice = 0.0
    best_ckpt_path    = None

    # ── Repeated Hold-out: 3회 반복 ────────────────────────────────────────────
    for i, seed in enumerate(CFG["seeds"]):
        print(f"\n{'='*60}")
        print(f"🔁 Repeated Hold-out  Run {i+1}/{CFG['n_repeats']}  (seed={seed})")
        print(f"{'='*60}")

        # group_id 기준 내부 train / 내부 val 분리
        groups = train_df["group_id"].unique()
        rng    = np.random.RandomState(seed)
        rng.shuffle(groups)
        split_n          = int(len(groups) * (1 - CFG["inner_val_ratio"]))
        inner_train_grps = set(groups[:split_n])
        inner_val_grps   = set(groups[split_n:])

        inner_train_df = train_df[train_df["group_id"].isin(inner_train_grps)]
        inner_val_df   = train_df[train_df["group_id"].isin(inner_val_grps)]

        best_dice, ckpt_path, epoch_log = train_single_run(
            inner_train_df, inner_val_df,
            run_id=i, seed=seed, device=device
        )

        run_results.append({
            "run": i, "seed": seed,
            "best_val_dice": best_dice,
            "ckpt_path": ckpt_path,
            "epoch_log": epoch_log,
        })

        if best_dice > best_overall_dice:
            best_overall_dice = best_dice
            best_ckpt_path    = ckpt_path

    # ── 결과 요약 ──────────────────────────────────────────────────────────────
    dice_list = [r["best_val_dice"] for r in run_results]
    print(f"\n{'='*60}")
    print(f"📊 Repeated Hold-out 결과 (3회)")
    for r in run_results:
        print(f"   Run {r['run']+1} (seed={r['seed']}): val Dice = {r['best_val_dice']:.4f}")
    print(f"   평균 Dice = {np.mean(dice_list):.4f}  ±  {np.std(dice_list):.4f}")
    print(f"   ✅ 최종 Stage 1 모델: {best_ckpt_path}  (Dice={best_overall_dice:.4f})")

    # ── 외부 validation 최종 평가 ──────────────────────────────────────────────
    print(f"\n🔍 외부 validation set ({len(val_df)}장) 최종 평가...")
    model = build_model().to(device)
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    criterion = BCEDiceLoss(lam=CFG["lam_dice"], pos_weight=CFG["pos_weight"]).to(device)

    val_ds = ForensicDataset(val_df, mode="val", target_size=CFG["target_size"])
    val_dl = DataLoader(val_ds, batch_size=CFG["batch_size"], shuffle=False,
                        num_workers=CFG["num_workers"])

    all_metrics = {"iou": [], "dice": [], "precision": [], "recall": [], "fp_ratio": []}

    model.eval()
    with torch.no_grad():
        for images, masks, labels in val_dl:
            images = images.to(device)
            preds  = torch.sigmoid(model(images)).cpu().numpy()
            gts    = masks.numpy()
            labs   = labels.numpy()

            for pred, gt, lab in zip(preds, gts, labs):
                m = compute_stage1_metrics(pred[0], gt[0])
                # Stage 1 localization 지표는 composite에서만 평균
                if lab == 1 and gt[0].sum() > 0:
                    all_metrics["iou"].append(m["iou"])
                    all_metrics["dice"].append(m["dice"])
                    all_metrics["precision"].append(m["precision"])
                    all_metrics["recall"].append(m["recall"])
                # real은 조작 영역이 없으므로 FP 면적 비율만 별도로 기록
                if lab == 0 and m["fp_area_ratio"] is not None:
                    all_metrics["fp_ratio"].append(m["fp_area_ratio"])

    def safe_mean(values):
        return float(np.mean(values)) if values else 0.0

    print("\n  외부 val 결과:")
    print(f"    Composite IoU       = {safe_mean(all_metrics['iou']):.4f}")
    print(f"    Composite Dice      = {safe_mean(all_metrics['dice']):.4f}")
    print(f"    Composite Precision = {safe_mean(all_metrics['precision']):.4f}")
    print(f"    Composite Recall    = {safe_mean(all_metrics['recall']):.4f}")
    print(f"    Real FP Area Ratio  = {safe_mean(all_metrics['fp_ratio']):.4f}")

    # ── 로그 저장 ──────────────────────────────────────────────────────────────
    log = {
        "run_results"       : run_results,
        "best_ckpt_path"    : best_ckpt_path,
        "best_overall_dice" : best_overall_dice,
        "mean_dice"         : float(np.mean(dice_list)),
        "std_dice"          : float(np.std(dice_list)),
        "val_metrics"       : {k: safe_mean(v) for k, v in all_metrics.items()},
    }
    with open(RESULT_PATH, "w") as f:
        json.dump(log, f, indent=2, default=str)

    print(f"\n✅ Stage 1 학습 완료. 로그: {RESULT_PATH}")
    print(f"   최종 사용 체크포인트: {best_ckpt_path}")
