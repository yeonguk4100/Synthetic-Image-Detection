# 합성 이미지 탐지 (Synthetic Image Detection)

이미지가 진짜인지 합성·조작된 것인지 분류하고, **어느 영역이 조작되었는지**까지 함께 제시하는 2단계 기반 시스템입니다. 단순 진짜/가짜 판별을 넘어 판단 근거를 시각적으로 보여주는 데 초점을 맞췄습니다.

## 결과 요약

| 지표 | 값 |
|------|-----|
| Test Accuracy | 0.7923 |
| Test F1-score | 0.7897 |
| Test Precision | 0.7997 |
| Test Recall | 0.7800 |
| 전체 데이터 | 30,000장 (진짜 15,000 + 합성 15,000) |

최적 설정: heatmap threshold = 0.6, top-k = 5, feature version = Version4, 분류 threshold = 0.40

## 시스템 구조

```
입력 이미지
   │
   ▼
[Stage 1] U-Net (EfficientNet-B4)   →  조작 의심 영역 heatmap 생성
   │
   ▼
[중간 처리] heatmap → feature 추출   →  밝기·색상·경계·texture 등 16개 feature
   │
   ▼
[Stage 2] LightGBM 분류기            →  진짜(real) / 합성(composite) 이진 분류
   │
   ▼
[설명] SHAP 분석                      →  판단 근거 feature 시각화
```

핵심 가설: Stage 1이 찾아낸 "조작 의심 위치"를 Stage 2의 힌트로 사용하면, 이미지만 보고 판별할 때보다 정확도와 설명 가능성이 함께 높아진다.

가장 기여도가 높았던 feature는 `max_suspicious_score` (SHAP 평균 절댓값 1.42)로, Stage 1 heatmap의 최대 confidence가 분류에 결정적이었습니다.

## 데이터셋

| 클래스 | 출처 | 수량 |
|--------|------|------|
| 진짜 (label 0) | COCO | 15,000 |
| 합성 (label 1) | DEFACTO | 15,000 |

split 구성: train 21,000 / validation 6,000 / test 3,000 (각 클래스 균형). `group_id` 기준으로 split 간 데이터 누수가 없도록 분리했습니다.

## 폴더 구조

```
synthetic-image-detection/
├─ src/                     # 코드 본체 (실행 순서대로)
│  ├─ step00_setup.py       # 라이브러리 설치 + Drive 마운트 + 폴더 생성
│  ├─ step01_data_prep.py   # zip 2개 → dataset.csv 생성·검증
│  ├─ step02_train_stage1.py# Stage 1 U-Net 학습 (Repeated Hold-out 3회)
│  ├─ step03_heatmap_features.py # heatmap 생성 + feature 추출
│  ├─ step04_train_stage2.py# Stage 2 LightGBM 학습 (5-fold CV)
│  ├─ step05_evaluate.py    # 최종 평가 + SHAP + 시각화
│  └─ utils.py              # 공통 모듈 (Dataset, Loss, feature 추출)
│
├─ notebooks/               # Colab 실행용 노트북
│  ├─ runner_share.ipynb    # 전체 파이프라인 실행 런처
│  ├─ runner_share_v1.ipynb # 개선 버전
│  ├─ stage1.ipynb          # Stage 1 실행
│  ├─ stage2.ipynb          # Stage 2 실행
│  ├─ demo_v2_02.ipynb      # 데모 (결과 시각화 포함)
│  └─ demo_v2_03.ipynb      # 데모 (결과 시각화 포함)
│
├─ docs/                    # 발표 자료
│  ├─ 발표대본.docx
│  ├─ 발표자료.pptx
│  └─ colab_실행기록.pdf
│
└─ results/                 # 결과물 (가벼운 것만)
   ├─ best_config.json      # 최적 설정
   ├─ test_metrics.json     # 최종 성능 지표
   ├─ cv_results.csv        # CV 탐색 결과
   ├─ shap_importance.csv   # feature 중요도
   ├─ stage1_training_log.json
   └─ visualizations/       # confusion matrix, SHAP, 샘플 시각화 PNG
```

## 실행 방법

이 코드는 **Google Colab 환경** 기준으로 작성되었습니다. 경로가 `/content/drive/MyDrive/forensic_project`로 고정되어 있으므로, Colab에서 아래 순서대로 실행하세요.

1. `step00_setup.py` — 라이브러리 설치, Drive 마운트, 폴더 구조 생성
2. 데이터 zip 2개(`defacto_15000_probe.zip`, `coco_real_15000.zip`)를 `MyDrive/forensic_uploads/`에 업로드
3. `step01_data_prep.py` — dataset.csv 생성
4. `step02_train_stage1.py` — Stage 1 학습 (체크포인트 생성)
5. `step03_heatmap_features.py` — heatmap + feature 생성
6. `step04_train_stage2.py` — Stage 2 학습
7. `step05_evaluate.py` — 최종 평가 + 시각화

`notebooks/runner_share_v1.ipynb`를 사용하면 위 과정을 한 번에 실행할 수 있습니다.

## 제외된 항목 안내

아래 항목은 **파일 용량이 커서** 이 저장소에서 제외되었습니다. 의도적으로 빠뜨린 것이 아니며, 위 실행 과정을 거치면 모두 다시 생성됩니다.

| 제외 항목 | 용량 | 재생성 방법 |
|-----------|------|-------------|
| 데이터셋 zip 2개 | 수 GB | 원본 데이터셋(COCO, DEFACTO) 별도 확보 |
| 모델 체크포인트 (`*.pth`) | 약 78MB × 3 | step02 실행 시 생성 |
| heatmap 30,000장 | 대용량 | step03 실행 시 생성 |
| feature CSV 6개 | 약 74MB | step03 실행 시 생성 |
| dataset.csv | 약 5MB | step01 실행 시 생성 |

> GitHub은 파일 1개당 100MB 제한이 있어, 대용량 모델·데이터는 일반적으로 저장소에 직접 올리지 않습니다.

## 기술 스택

PyTorch, segmentation-models-pytorch (U-Net), LightGBM, SHAP, scikit-learn, albumentations, OpenCV, scikit-image
