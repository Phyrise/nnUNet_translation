# sCT U-Net (MONAI 2.5D) — 계획 및 진행상황

CBCT → sCT (synthetic CT) 합성용 2.5D U-Net. 분할용 U-Net을 **회귀(image-to-image)** 로 개조한 fork.
베이스: MONAI `UNet`. 데이터: `c:/Users/sera.QUALITICS2/nnunet/data/raw/Dataset050_CBCT` (입력) / `Dataset051_CT` (타겟), 케이스별로 같은 좌표계 정합 완료 (14 케이스).

---

## 1. 설계 요약

| 항목 | 값 |
|---|---|
| Task | CBCT → CT 합성 (paired, voxel-wise regression) |
| 입력 | CBCT 슬라이스 z 와 인접 z-1, z+1 → **3채널 2D** (2.5D) |
| 출력 | 동일 위치 CT 슬라이스 z (1채널 2D, HU) |
| Backbone | `monai.networks.nets.UNet` (spatial_dims=2, in_channels=3, out_channels=1) |
| 출력 활성화 | 없음 (linear) — HU 회귀 |
| Loss | L1 (MAE). 옵션: L1 + 0.1·SSIM |
| Optimizer | Adam, lr=1e-4, weight_decay=1e-5 |
| Scheduler | CosineAnnealingLR |
| 정규화 | CBCT: per-volume z-score (mask 안). CT: HU 클립 [-1000, 2000] → [-1, 1] 선형 매핑 |
| 데이터 분할 | 14 케이스 → train 11 / val 3 (case-level split, 환자 누수 방지) |
| Augmentation | RandFlipd(좌우), RandAffined(회전 ±10°, 스케일 0.9–1.1), RandGaussianNoised |
| Patch | 256×256 random crop (학습), 추론 시 sliding window 256×256 stride 128 |
| 학습 epochs | 200 (early stop on val MAE) |
| 평가 | MAE(HU), PSNR, SSIM, 뼈/연조직 마스크별 MAE |

분할 U-Net과의 차이 (개조 포인트):

1. `out_channels = 1`, softmax/argmax 제거
2. 손실: CrossEntropy/Dice → **L1**
3. 데이터셋: (image, mask) → **(CBCT, CT)** — 양쪽 다 NIfTI 볼륨, GT가 연속값
4. CT는 HU 정규화 + denorm 필요 (학습은 [-1,1], 저장은 HU 복원)
5. 추론은 슬라이스 단위 → 볼륨 재조립 (NIfTI affine 보존)

### 1.1. 베이스 MONAI 선택 이유

- **의료영상 I/O 빌트인**: `LoadImaged`, `Spacingd`, `Orientationd` 등 NIfTI affine·spacing·orientation을 그대로 처리. dictionary transform으로 CBCT/CT 페어에 동일 augmentation을 동시에 적용 가능 (회귀 task 필수).
- **UNet이 분할/회귀 양쪽 호환**: `monai.networks.nets.UNet`은 `out_channels=1` + 마지막 활성화 제거만으로 회귀 전환. `spatial_dims=2`로 2D/2.5D 스위칭도 한 줄.
- **추론·평가 유틸**: `sliding_window_inference`(256×256 patch → 볼륨 복원), `SSIMMetric`/`PSNRMetric` 빌트인.
- **소규모(14 케이스)에 적합**: 라이브러리 형태라 학습 루프를 명시적으로 짤 수 있어 디버깅·시각 검수가 쉬움.

대안 비교:

| 후보 | 채택 안 한 이유 |
|---|---|
| 순수 PyTorch | NIfTI I/O·페어 augmentation·sliding window를 직접 구현해야 함 |
| nnU-Net fork | 분할 가정(softmax/Dice/argmax)이 깊이 박혀 회귀 개조가 침습적. 자동 planner도 회귀에 부적합 |

---

## 2. 디렉토리 구조

```
sct_unet/
├── PLAN.md                  ← (이 문서)
├── README.md                ← 빠른 실행 가이드
├── requirements.txt
├── configs/
│   └── default.yaml         ← 경로/하이퍼 파라미터
├── src/
│   ├── __init__.py
│   ├── data.py              ← 2.5D paired dataset (NIfTI → slices)
│   ├── model.py             ← MONAI UNet 래퍼
│   ├── losses.py            ← L1 (+ SSIM 옵션)
│   ├── utils.py             ← 정규화/역정규화, 시드, 로거
│   ├── train.py             ← 학습 엔트리
│   └── infer.py             ← 추론 엔트리 (슬라이스 → 볼륨 복원)
├── scripts/
│   ├── train.sh / train.bat
│   └── infer.sh / infer.bat
└── outputs/                 ← 체크포인트, 로그, 예측 NIfTI (런타임 생성)
```

GPU 머신 이전 시 이 폴더 + 데이터(또는 `configs/default.yaml`의 `data_root`만 수정)만 옮기면 됨. nnUNet 레포는 의존하지 않음.

---

## 3. 진행 상황

- [x] 데이터 검증: CBCT/CT shape 일치, ~310×307×85, 1×1×3mm, int16. 페어링 OK.
- [x] 디렉토리 생성 (`sct_unet/`)
- [x] PLAN.md 작성
- [x] `requirements.txt` 작성
- [x] `configs/default.yaml` 작성
- [x] `src/utils.py` (정규화, 시드, 로거)
- [x] `src/data.py` (2.5D paired dataset)
- [x] `src/model.py` (MONAI UNet)
- [x] `src/losses.py` (L1 + 옵션 SSIM)
- [x] `src/train.py`
- [x] `src/infer.py`
- [x] `scripts/train.{sh,bat}`, `scripts/infer.{sh,bat}`
- [x] `README.md` (Quickstart)
- [x] 모든 src 모듈 syntax 검사 통과 (CPU 환경, import 검증은 GPU 머신에서)
- [ ] **GPU 머신 smoke test** (CUDA 인식, `pip install -r requirements.txt`, 1 epoch 학습, val 케이스 1개 추론) ← 사용자 직접 실행 예정

---

## 4. 빠른 실행 흐름 (생성 후)

```bash
# GPU 머신에서
cd sct_unet
pip install -r requirements.txt

# 데이터 경로가 다르면 configs/default.yaml의 data_root 수정
python -m src.train --config configs/default.yaml
python -m src.infer --config configs/default.yaml --ckpt outputs/best.pth \
    --input /path/to/CBCT.nii.gz --output /path/to/sCT.nii.gz
```

---

## 5. 결정된 사항 / 미결 사항

**결정**
- Base: MONAI UNet (2D, 채널 3 입력)
- Loss: L1 우선, SSIM은 옵션 플래그
- 정규화: CT는 HU [-1000, 2000] 선형 → [-1, 1]
- Split: case-level (환자 단위) 11/3

**미결 (필요 시 학습 후 조정)**
- AFP loss / perceptual loss 추가 여부
- 5-fold CV 적용 여부 (지금은 단일 split)
- 슬라이스 두께 비등방(1×1×3mm) → z 축 augmentation 비활성 (현재 계획대로)

---

## 6. 위험/주의

- 14 케이스로는 일반화 한계 — val MAE만으로 판단 말고 시각 검수 필수
- HU 클립 범위 고정([-1000, 2000])이라 임플란트/금속이 있으면 saturation 발생 → 케이스 검수 필요
- CBCT z-score 정규화는 air 비율 영향 큼 → foreground mask 기반 정규화로 후속 개선 가능
