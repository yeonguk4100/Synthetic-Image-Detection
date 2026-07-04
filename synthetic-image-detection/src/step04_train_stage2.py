"""
Step 04 — Stage 2 LightGBM 학습 (5-fold Cross-Validation)

흐름:
  1. threshold × top-k × feature 조합 × hyperparameter 후보를 5-fold CV로 탐색
  2. 평균 F1이 가장 높은 설정 선택
  3. 선택된 설정으로 train 전체 재학습
  4. validation으로 최종 사전 평가 + 분류 threshold 미세 조정
  5. 최적 설정·모델 저장
"""

import json
import os
import sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_STRATIFIED_GROUP = True
except Exception:
    from sklearn.model_selection import GroupKFold
    HAS_STRATIFIED_GROUP = False

sys.path.insert(0, "/content/drive/MyDrive/forensic_project")
from utils import FEATURE_VERSIONS

BASE_DIR = "/content/drive/MyDrive/forensic_project"
FEATURE_DIR = f"{BASE_DIR}/features"
MODEL_DIR = f"{BASE_DIR}/results"
os.makedirs(MODEL_DIR, exist_ok=True)

THRESHOLDS = [0.4, 0.5, 0.6]
TOP_KS = [3, 5]
FEAT_VERSIONS = list(FEATURE_VERSIONS.keys())

# 전체 grid는 너무 오래 걸리므로 발표/과제용으로 대표 3세트 사용
LGBM_PARAMS_LIST = [
    {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 6,
     "min_child_samples": 20, "num_leaves": 63, "verbose": -1,
     "random_state": 42, "n_jobs": -1},
    {"n_estimators": 500, "learning_rate": 0.01, "max_depth": 8,
     "min_child_samples": 10, "num_leaves": 127, "verbose": -1,
     "random_state": 42, "n_jobs": -1},
    {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 4,
     "min_child_samples": 50, "num_leaves": 15, "verbose": -1,
     "random_state": 42, "n_jobs": -1},
]

N_FOLDS = 5


def get_cv_splitter():
    if HAS_STRATIFIED_GROUP:
        return StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    return GroupKFold(n_splits=N_FOLDS)


def validate_feature_cols(feat_df, feat_cols):
    missing = [c for c in feat_cols if c not in feat_df.columns]
    if missing:
        raise ValueError(f"feature CSV에 필요한 컬럼이 없습니다: {missing}")


def run_cv(feat_df, feat_cols, params):
    X = feat_df[feat_cols].values
    y = feat_df["label"].astype(int).values
    groups = feat_df["group_id"].values

    splitter = get_cv_splitter()
    f1s, accs = [], []

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(X, y, groups)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        clf = lgb.LGBMClassifier(**params)
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=0)],
        )
        y_pred = clf.predict(X_va)
        f1s.append(f1_score(y_va, y_pred, zero_division=0))
        accs.append(accuracy_score(y_va, y_pred))

    return float(np.mean(f1s)), float(np.mean(accs))


print("5-fold CV 탐색 시작")
print(f"CV splitter: {'StratifiedGroupKFold' if HAS_STRATIFIED_GROUP else 'GroupKFold'}")
print(f"조합 수: {len(THRESHOLDS)} threshold × {len(TOP_KS)} top_k × "
      f"{len(FEAT_VERSIONS)} feature 버전 × {len(LGBM_PARAMS_LIST)} param 셋")

all_results = []

for thr in THRESHOLDS:
    for k in TOP_KS:
        combo_name = f"thr{int(thr * 10)}_k{k}"
        feat_path = f"{FEATURE_DIR}/features_{combo_name}.csv"

        if not os.path.exists(feat_path):
            print(f"파일 없음, skip: {feat_path}")
            continue

        feat_df = pd.read_csv(feat_path)
        train_feat = feat_df[feat_df["split"] == "train"].reset_index(drop=True)

        if train_feat.empty:
            raise RuntimeError(f"train feature가 비어 있습니다: {feat_path}")

        for version_name in FEAT_VERSIONS:
            feat_cols = FEATURE_VERSIONS[version_name]
            validate_feature_cols(train_feat, feat_cols)

            for pi, params in enumerate(LGBM_PARAMS_LIST):
                mean_f1, mean_acc = run_cv(train_feat, feat_cols, params)
                result = {
                    "threshold": thr,
                    "top_k": k,
                    "feat_version": version_name,
                    "param_idx": pi,
                    "mean_f1": mean_f1,
                    "mean_acc": mean_acc,
                }
                all_results.append(result)
                print(f"thr={thr} k={k} {version_name:10s} params#{pi} -> F1={mean_f1:.4f}, Acc={mean_acc:.4f}")

if not all_results:
    raise RuntimeError("CV 결과가 없습니다. step03 feature CSV 생성 여부를 확인하세요.")

results_df = pd.DataFrame(all_results).sort_values("mean_f1", ascending=False)
cv_path = f"{MODEL_DIR}/cv_results.csv"
results_df.to_csv(cv_path, index=False, encoding="utf-8-sig")
print(f"\nCV 탐색 완료. 결과 저장: {cv_path}")

best = results_df.iloc[0]
print("\n최적 설정")
print(f"threshold    = {best['threshold']}")
print(f"top_k        = {int(best['top_k'])}")
print(f"feat_version = {best['feat_version']}")
print(f"param_idx    = {int(best['param_idx'])}")
print(f"CV mean F1   = {best['mean_f1']:.4f}")

BEST_THR = float(best["threshold"])
BEST_K = int(best["top_k"])
BEST_VERSION = str(best["feat_version"])
BEST_PARAMS = LGBM_PARAMS_LIST[int(best["param_idx"])]
BEST_COLS = FEATURE_VERSIONS[BEST_VERSION]

feat_path = f"{FEATURE_DIR}/features_thr{int(BEST_THR * 10)}_k{BEST_K}.csv"
feat_df = pd.read_csv(feat_path)
train_feat = feat_df[feat_df["split"] == "train"].reset_index(drop=True)
val_feat = feat_df[feat_df["split"] == "validation"].reset_index(drop=True)

X_train = train_feat[BEST_COLS].values
y_train = train_feat["label"].astype(int).values
X_val = val_feat[BEST_COLS].values
y_val = val_feat["label"].astype(int).values

print("\n전체 train으로 최종 LightGBM 학습 중")
final_clf = lgb.LGBMClassifier(**BEST_PARAMS)
final_clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="binary_logloss",
    callbacks=[lgb.early_stopping(50, verbose=True),
               lgb.log_evaluation(period=50)],
)
print("재학습 완료")

print("\nValidation 최종 사전 평가 + 분류 threshold 미세 조정")
val_probs = final_clf.predict_proba(X_val)[:, 1]

best_thr_f1, best_val_thr = -1.0, 0.5
for t in np.arange(0.30, 0.701, 0.05):
    y_pred_t = (val_probs >= t).astype(int)
    f1 = f1_score(y_val, y_pred_t, zero_division=0)
    if f1 > best_thr_f1:
        best_thr_f1 = f1
        best_val_thr = float(t)

y_pred_final = (val_probs >= best_val_thr).astype(int)
print(f"최적 분류 threshold: {best_val_thr:.2f}")
print("Validation 성능:")
print(classification_report(y_val, y_pred_final, target_names=["real", "composite"], zero_division=0))

val_pred_path = f"{MODEL_DIR}/validation_predictions.csv"
val_out = val_feat[["image_path", "label", "split", "group_id"]].copy()
val_out["prob_composite"] = val_probs
val_out["pred_label"] = y_pred_final
val_out.to_csv(val_pred_path, index=False, encoding="utf-8-sig")
print(f"validation 예측 저장: {val_pred_path}")

model_path = f"{MODEL_DIR}/stage2_lgbm.pkl"
joblib.dump(final_clf, model_path)
print(f"모델 저장: {model_path}")

best_config = {
    "threshold": BEST_THR,
    "top_k": BEST_K,
    "feat_version": BEST_VERSION,
    "feat_cols": BEST_COLS,
    "lgbm_params": BEST_PARAMS,
    "classify_threshold": float(best_val_thr),
    "validation_f1_at_classify_threshold": float(best_thr_f1),
    "cv_mean_f1": float(best["mean_f1"]),
}
config_path = f"{MODEL_DIR}/best_config.json"
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(best_config, f, indent=2, ensure_ascii=False)

print(f"최적 설정 저장: {config_path}")
print("\n다음 step에서 사용할 설정")
print(f"분류 threshold = {best_val_thr:.2f}")
print(f"feature 버전   = {BEST_VERSION}")
print(f"후처리 설정    = threshold={BEST_THR}, top_k={BEST_K}")
