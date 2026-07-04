# 합성 이미지 탐지 (Synthetic Image Detection)

이미지가 진짜인지 합성·조작된 것인지 분류하고, **어느 영역이 조작되었는지**까지 함께 제시하는 2단계 기반 시스템입니다. 단순 진짜/가짜 판별을 넘어 판단 근거를 시각적으로 보여주는 데 초점을 맞췄습니다.

<img width="1937" height="643" alt="조작 탐지 예시" src="https://github.com/user-attachments/assets/5329dd97-30f4-422c-8926-27cc5f880ce9" />

*합성 이미지 탐지 예시 — Stage 1 heatmap이 조작 의심 영역(빨간색)을 짚어내고, Stage 2가 composite(prob=0.966)로 분류*

## 결과 요약

| 지표 | 값 |
|------|-----|
| Test Accuracy | 0.7923 |
| Test F1-score | 0.7897 |
| Test Precision | 0.7997 |
| Test Recall | 0.7800 |
| 전체 데이터 | 30,000장 (진짜 15,000 + 합성 15,000) |

최적 설정: heatmap threshold = 0.6, top-k = 5, feature version = Version4, 분류 threshold = 0.40

| Test (3,000장) | Validation (6,000장) |
|:---:|:---:|
| <img width="465" alt="Test Confusion Matrix" src="https://github.com/user-attachments/assets/e6ce9c17-43c2-45b7-86ff-f27b01f1c87b" /> | <img width="465" alt="Validation Confusion Matrix" src="https://github.com/user-attachments/assets/a33d58fd-c956-4608-8f8e-f5c0418de29b" /> |

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

설계 원칙:
- 한 모델이 위치(mask)와 분류(label)를 동시에 맞추면 loss trade-off가 발생하므로, Stage 1은 위치 추정, Stage 2는 최종 분류로 역할을 분리
- inference 입력은 **이미지 1장만** 사용 (mask는 Stage 1 학습·평가용 GT로만 사용, test 입력에 미포함)
- tabular feature 기반 LightGBM을 채택해 SHAP으로 판단 근거를 설명 가능하게 설계

## 예측 예시

| 성공 사례 | 실패 사례 |
|:---:|:---:|
| <img width="1937" alt="composite 탐지 성공" src="https://github.com/user-attachments/assets/5329dd97-30f4-422c-8926-27cc5f880ce9" /> | <img width="1936" alt="composite 탐지 실패" src="https://github.com/user-attachments/assets/d20728ae-51cb-4fd7-8fe9-fef710468e56" /> |
| **합성 이미지 탐지 성공** — heatmap이 삽입 영역을 강하게 포착 (prob=0.966, max_score=0.85) | **합성 이미지 놓침** — 조작 영역의 경계 불일치가 약해 heatmap 반응이 낮음 (prob=0.148, max_score=0.31) |
| <img width="1936" alt="real 판별 성공" src="https://github.com/user-attachments/assets/4bff679d-3b01-4cd7-a1e1-17fd45565f18" /> | <img width="1937" alt="real 오탐" src="https://github.com/user-attachments/assets/0ddcacae-af38-4148-8075-539b60f3cd7d" /> |
| **진짜 이미지 정상 판별** — heatmap 전반이 조용함 (prob=0.062, max_score=0.29) | **진짜 이미지 오탐** — 자연스러운 고대비 영역을 조작으로 오인 (prob=0.735, max_score=0.77) |

실패 사례 두 건 모두 `max_suspicious_score`가 분류를 좌우했음을 보여줍니다. Stage 1 heatmap 품질이 곧 최종 성능의 상한이라는 구조적 특성이 드러납니다.

## 판단 근거 분석 (SHAP)

<img width="1184" height="821" alt="SHAP Feature Importance" src="https://github.com/user-attachments/assets/3c4a0428-acfe-46e2-8fd1-adf70f0608cf" />

가장 기여도가 높았던 feature는 `max_suspicious_score` (SHAP 평균 절댓값 1.42)로, Stage 1 heatmap의 최대 confidence가 분류에 결정적이었습니다. 이는 "heatmap을 분류 feature로 쓰면 유효하다"는 시작 가설을 뒷받침합니다.

<img width="700" alt="SHAP Summary Plot" src="https://github.com/user-attachments/assets/b99e1e3a-5b75-478f-b632-07830c08fdb3" />

*beeswarm plot — 각 feature 값의 높낮이(색)가 composite/real 판단(x축)에 미친 방향과 크기*

## 데이터셋

| 클래스 | 출처 | 수량 |
|--------|------|------|
| 진짜 (label 0) | COCO | 15,000 |
| 합성 (label 1) | DEFACTO | 15,000 |

split 구성: train 21,000 / validation 6,000 / test 3,000 (각 클래스 균형). `group_id` 기준으로 split 간 데이터 누수가 없도록 분리했습니다.

## 검증 전략

| 단계 | 방법 |
|------|------|
| Stage 1 (U-Net) | Repeated Hold-out 3회 |
| Stage 2 (LightGBM) | 5-fold Cross Validation |
| 최종 threshold 선택 | validation 기준 탐색 후 test 1회 평가 |

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

## 한계와 개선 방향

- **JPG 압축 아티팩트**: 저장 포맷에 따른 mask 경계 손실 → mask closing 연산으로 GT 품질 보완
- **입력 해상도**: 현재보다 큰 512 해상도 학습으로 미세한 경계 불일치 포착력 향상
- **Stage 1 아키텍처**: U-Net → U-Net++ 등 skip connection 강화 구조 실험
- **클래스 불균형 대응**: pixel 단위 positive 비율이 낮은 문제에 `pos_weight` 조정 적용
- **구조적 한계**: 실패 사례에서 확인했듯 Stage 2 성능이 Stage 1 heatmap 품질에 종속됨 → Stage 1 개선이 최우선

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
