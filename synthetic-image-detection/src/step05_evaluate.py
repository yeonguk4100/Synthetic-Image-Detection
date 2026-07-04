"""
Step 05 — 최종 평가 + SHAP 분석 + 시각화

출력:
  - test_metrics.json
  - predictions_test.csv
  - confusion_matrix.png
  - shap_summary.png
  - shap_importance.png
  - shap_waterfall_*.png
  - sample_*.png
"""

import json
import os
import sys

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, "/content/drive/MyDrive/forensic_project")
from utils import letterbox_resize

BASE_DIR = "/content/drive/MyDrive/forensic_project"
FEATURE_DIR = f"{BASE_DIR}/features"
MODEL_DIR = f"{BASE_DIR}/results"
VIZ_DIR = f"{BASE_DIR}/results/visualizations"
CSV_PATH = f"{BASE_DIR}/data/dataset.csv"
os.makedirs(VIZ_DIR, exist_ok=True)

with open(f"{MODEL_DIR}/best_config.json", encoding="utf-8") as f:
    cfg = json.load(f)

BEST_THR = float(cfg["threshold"])
BEST_K = int(cfg["top_k"])
BEST_VERSION = cfg["feat_version"]
FEAT_COLS = cfg["feat_cols"]
CLASSIFY_THR = float(cfg["classify_threshold"])
MIN_AREA = 100

print("최적 설정")
for k, v in cfg.items():
    print(f"  {k}: {v}")

clf = joblib.load(f"{MODEL_DIR}/stage2_lgbm.pkl")
feature_path = f"{FEATURE_DIR}/features_thr{int(BEST_THR * 10)}_k{BEST_K}.csv"
feat_df = pd.read_csv(feature_path)

# 혹시 오래된 feature CSV에 heatmap_path/mask_path가 없다면 dataset.csv에서 보강
main_df = pd.read_csv(CSV_PATH)
needed_paths = ["mask_path", "heatmap_path", "dataset_name", "manipulation_type"]
missing_path_cols = [c for c in needed_paths if c not in feat_df.columns]
if missing_path_cols:
    merge_cols = ["image_path"] + [c for c in needed_paths if c in main_df.columns]
    feat_df = feat_df.merge(main_df[merge_cols], on="image_path", how="left")

missing_features = [c for c in FEAT_COLS if c not in feat_df.columns]
if missing_features:
    raise ValueError(f"feature CSV에 필요한 컬럼이 없습니다: {missing_features}")

test_df = feat_df[feat_df["split"] == "test"].reset_index(drop=True)
if test_df.empty:
    raise RuntimeError("test feature가 비어 있습니다.")

X_test = test_df[FEAT_COLS].values
y_test = test_df["label"].astype(int).values

print("\n" + "=" * 60)
print(f"Test Set 최종 평가 ({len(test_df)}장)")
print("=" * 60)

y_probs = clf.predict_proba(X_test)[:, 1]
y_pred = (y_probs >= CLASSIFY_THR).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print(f"Accuracy  = {acc:.4f}")
print(f"Precision = {prec:.4f}")
print(f"Recall    = {rec:.4f}")
print(f"F1-score  = {f1:.4f}")
print("\nConfusion Matrix (행=실제, 열=예측)")
print("              real  composite")
print(f"real       {cm[0, 0]:6d}  {cm[0, 1]:6d}")
print(f"composite  {cm[1, 0]:6d}  {cm[1, 1]:6d}")
print("\n" + classification_report(y_test, y_pred, target_names=["real", "composite"], zero_division=0))

metrics = {
    "accuracy": float(acc),
    "precision": float(prec),
    "recall": float(rec),
    "f1": float(f1),
    "confusion_matrix": cm.tolist(),
    "classify_threshold": CLASSIFY_THR,
    "feature_version": BEST_VERSION,
    "heatmap_threshold": BEST_THR,
    "top_k": BEST_K,
}
with open(f"{MODEL_DIR}/test_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

pred_out = test_df[["image_path", "label", "split", "group_id"]].copy()
pred_out["prob_composite"] = y_probs
pred_out["pred_label"] = y_pred
pred_out["correct"] = pred_out["label"].astype(int) == pred_out["pred_label"].astype(int)
pred_path = f"{MODEL_DIR}/predictions_test.csv"
pred_out.to_csv(pred_path, index=False, encoding="utf-8-sig")
print(f"test 예측 저장: {pred_path}")

# Confusion Matrix 시각화 — matplotlib만 사용
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm)
ax.set_xticks([0, 1])
ax.set_xticklabels(["real", "composite"])
ax.set_yticks([0, 1])
ax.set_yticklabels(["real", "composite"])
ax.set_xlabel("예측")
ax.set_ylabel("실제")
ax.set_title(f"Confusion Matrix (F1={f1:.3f})")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center")
fig.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/confusion_matrix.png", dpi=150)
plt.close()
print(f"Confusion Matrix 저장: {VIZ_DIR}/confusion_matrix.png")

print("\nSHAP 분석 중")
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

if isinstance(shap_values, list):
    sv_composite = shap_values[1]
else:
    sv_composite = shap_values

expected_value = explainer.expected_value
if isinstance(expected_value, list):
    base_value = expected_value[1]
elif isinstance(expected_value, np.ndarray) and expected_value.ndim > 0 and len(expected_value) > 1:
    base_value = expected_value[1]
else:
    base_value = expected_value

plt.figure(figsize=(9, 6))
shap.summary_plot(sv_composite, X_test, feature_names=FEAT_COLS, show=False, plot_type="dot")
plt.title("SHAP Summary Plot — composite 판단 기여도")
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("SHAP Summary Plot 저장")

mean_abs_shap = np.abs(sv_composite).mean(0)
fi_df = pd.DataFrame({"feature": FEAT_COLS, "mean_abs_shap": mean_abs_shap})
fi_df = fi_df.sort_values("mean_abs_shap", ascending=True)
fi_df.to_csv(f"{MODEL_DIR}/shap_importance.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(8, max(4, len(FEAT_COLS) * 0.35)))
plt.barh(fi_df["feature"], fi_df["mean_abs_shap"])
plt.xlabel("Mean |SHAP value|")
plt.title("Feature 중요도 (SHAP 기반)")
plt.tight_layout()
plt.savefig(f"{VIZ_DIR}/shap_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("SHAP Importance 저장")

# Waterfall Plot: 올바르게 composite로 맞춘 샘플 우선, 없으면 composite 확률 높은 샘플
composite_idx = np.where((y_pred == 1) & (y_test == 1))[0]
if len(composite_idx) == 0:
    composite_idx = np.argsort(y_probs)[::-1][:3]
else:
    composite_idx = composite_idx[:3]

for rank, idx in enumerate(composite_idx):
    exp = shap.Explanation(
        values=sv_composite[idx],
        base_values=base_value,
        data=X_test[idx],
        feature_names=FEAT_COLS,
    )
    plt.figure(figsize=(8, 5))
    shap.plots.waterfall(exp, show=False)
    plt.title(f"SHAP Waterfall #{rank + 1} (composite 확률={y_probs[idx]:.2f})")
    plt.tight_layout()
    plt.savefig(f"{VIZ_DIR}/shap_waterfall_{rank + 1}.png", dpi=150, bbox_inches="tight")
    plt.close()
print("SHAP Waterfall Plot 저장")

print("\n최종 출력 시각화 중")


def load_image_letterbox(path, target=256):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"이미지 없음: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return letterbox_resize(img, target)


def load_mask_letterbox(path, target=256):
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros((target, target), dtype=np.uint8)
    dummy = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    _, mask_resized = letterbox_resize(dummy, target, mask=mask)
    return (mask_resized > 127).astype(np.uint8)




def load_heatmap(path):
    path = str(path)
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32)
    hm = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if hm is None:
        raise FileNotFoundError(f"heatmap 없음: {path}")
    return hm.astype(np.float32) / 255.0

def visualize_prediction(image_rgb, heatmap, gt_mask, pred_label, prob,
                         shap_arr, feat_cols, threshold=BEST_THR,
                         top_k=BEST_K, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    label_str = "composite" if pred_label == 1 else "real"
    fig.suptitle(f"예측: {label_str}  (composite 확률={prob:.2f})", fontsize=13, fontweight="bold")

    binary = (heatmap > threshold).astype(np.uint8)
    num_l, label_map, stats, _ = cv2.connectedComponentsWithStats(binary)
    valid = [(int(stats[l, cv2.CC_STAT_AREA]), stats[l, :4]) for l in range(1, num_l)
             if int(stats[l, cv2.CC_STAT_AREA]) >= MIN_AREA]
    valid.sort(key=lambda x: x[0], reverse=True)

    img_draw = image_rgb.copy()
    for _, (x, y, w, h) in valid[:top_k]:
        cv2.rectangle(img_draw, (x, y), (x + w, y + h), (255, 50, 50), 2)

    axes[0].imshow(img_draw)
    axes[0].set_title("원본 + 의심 영역 bounding box")
    axes[0].axis("off")

    axes[1].imshow(image_rgb)
    im = axes[1].imshow(heatmap, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    axes[1].set_title(f"Stage 1 Heatmap (threshold={threshold})")
    axes[1].axis("off")

    top5_idx = np.argsort(np.abs(shap_arr))[::-1][:5]
    top5_vals = shap_arr[top5_idx]
    top5_feat = [feat_cols[i] for i in top5_idx]
    axes[2].barh(range(len(top5_idx)), top5_vals[::-1])
    axes[2].set_yticks(range(len(top5_idx)))
    axes[2].set_yticklabels(top5_feat[::-1], fontsize=9)
    axes[2].axvline(0, linewidth=0.8)
    axes[2].set_xlabel("SHAP value")
    axes[2].set_title("SHAP 기여도 Top-5")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()


# composite 정답/예측 3개 + real 정답/예측 3개
real_idx = np.where((y_pred == 0) & (y_test == 0))[0][:3]
targets = list(composite_idx[:3]) + list(real_idx)

for rank, idx in enumerate(targets):
    row = test_df.iloc[int(idx)]
    img = load_image_letterbox(row["image_path"], 256)
    heatmap = load_heatmap(row["heatmap_path"])
    gt_np = load_mask_letterbox(row["mask_path"], 256)

    label = "composite" if y_pred[idx] == 1 else "real"
    save_p = f"{VIZ_DIR}/sample_{rank + 1:02d}_{label}.png"

    visualize_prediction(
        image_rgb=img,
        heatmap=heatmap,
        gt_mask=gt_np,
        pred_label=int(y_pred[idx]),
        prob=float(y_probs[idx]),
        shap_arr=sv_composite[idx],
        feat_cols=FEAT_COLS,
        save_path=save_p,
    )
    print(f"시각화 저장: {save_p}")

print("\n" + "=" * 60)
print("최종 결과 요약")
print("=" * 60)
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-score  : {f1:.4f}")
print("\nSHAP 상위 3 feature:")
fi_sorted = sorted(zip(FEAT_COLS, mean_abs_shap), key=lambda x: x[1], reverse=True)
for feat, val in fi_sorted[:3]:
    print(f"  {feat:<30s}: {val:.4f}")
print(f"\n시각화 저장 위치: {VIZ_DIR}/")
print("=" * 60)
