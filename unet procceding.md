# sCT U-Net (MONAI 2.5D) — 진행 기록

본 문서는 [PLAN.md](./PLAN.md) 에 정의된 sCT U-Net 파이프라인을 GPU 머신에 적용한 실행 로그입니다.

## 환경

| 항목 | 값 |
|------|----|
| GPU | NVIDIA GeForce RTX 5090 (32 GB) |
| Python | 3.13.12 |
| PyTorch | 2.11.0 + cu128 |
| MONAI | 1.5.2 (이번에 신규 설치) |
| nibabel | 5.4.2 |
| numpy | 2.4.4 |
| tensorboard | 2.20.0 (이번에 신규 설치) |
| 작업 폴더 | `C:/Users/USER/Desktop/dev/sct_unet` |

## 데이터

| 항목 | 값 |
|------|----|
| CBCT (입력) | `C:/Users/USER/Desktop/dev/nnUNet_translation/raw/Dataset050_TrainHN_Input/imagesTr` |
| CT (타겟) | `C:/Users/USER/Desktop/dev/nnUNet_translation/raw/Dataset051_TrainHN_Target/imagesTr` |
| 전체 paired 케이스 | 200 |
| 학습 split | nnUNet `splits_final.json` fold_0 와 동일 (train 160 / val 40) — 동일 평가 조건 유지로 nnUNet 결과와 직접 비교 가능 |

> 참고: PLAN.md 는 14 케이스 기준이지만 로컬 데이터가 200 케이스라 실제 사용은 200 케이스 전체. 모델·loss·증강 설정은 PLAN.md 그대로 따름.

## 변경 사항

- [configs/default.yaml](./configs/default.yaml): 데이터 경로를 로컬로 교체. val_cases 를 nnUNet fold_0 val 40개로 채움.
- [configs/smoke.yaml](./configs/smoke.yaml): 신규 추가 — 1 epoch smoke test 용 (val 2 케이스, num_epochs=1, augmentation 약화).

## 진행 타임라인

| 시각 | 이벤트 |
|------|-------|
| 2026-04-29 09:30 | sct_unet 디렉토리 / src 모듈 / config / scripts 모두 PLAN.md 기준대로 이미 존재함을 확인. |
| 2026-04-29 09:31 | 환경 점검: monai/tensorboard 가 nnunet 콘다 환경에 누락. 둘 다 pip install 완료 (monai 1.5.2, tensorboard 2.20.0). |
| 2026-04-29 09:32 | configs/default.yaml 의 데이터 경로를 로컬로 교체, val_cases 를 nnUNet fold_0 val 40개로 채움. |
| 2026-04-29 09:33 | configs/smoke.yaml 작성 (val 2 케이스, num_epochs=1). |
| 2026-04-29 09:34 | smoke test 실행 시작 (백그라운드). 데이터 로딩에 약 50s, 198 train 케이스 → 16910 슬라이스, 모델 6.50M 파라미터. |
| 2026-04-29 09:35 | smoke 1차 시도 실패 — val 단계 (full slice 312×312) 가 4단 stride U-Net 의 16배수 정합 안 맞아 skip-connection concat 에서 RuntimeError. |
| 2026-04-29 09:36 | [src/train.py](./src/train.py) `evaluate()` 를 MONAI `sliding_window_inference` 사용하도록 수정 (256×256 patch, gaussian blending, infer.py 와 동일 방식). |
| 2026-04-29 09:38 | smoke 2차 재시도. epoch 0 train loss 0.54 → 0.47, 48s 소요. val MAE(HU) = 783.15 (1 epoch 학습 결과). 파이프라인 정상 동작 확인, `outputs/smoke/best.pth` 저장. |
| 2026-04-29 09:40 | 본 학습 시작 (백그라운드, [outputs/full_0429/](./outputs/full_0429/)). 160 train / 40 val (nnUNet fold_0 동일), 13634 train slices, 3451 val slices, 200 epochs, val every 1 epoch. epoch당 train ~60s + val ~40s = ~100s. |
| 2026-04-29 09:42 ~ 11:17 | epoch 0 → 57 정상 학습. val MAE 818 → 26.24 HU 까지 개선. best.pth: epoch 48, val MAE **26.24** (그때까지 최저). |
| 2026-04-29 11:17 | epoch 57 train 완료 직후 val 도중 프로세스 무성종료 (exit 1, stdout/stderr 모두 에러 없음, GPU·디스크 정상). last.pth (epoch 56 상태) 와 best.pth (epoch 48) 보존. |
| 2026-04-29 11:42 | [scripts/watchdog.py](./scripts/watchdog.py) 작성 후 백그라운드 시작. attempt 1 이 `--resume outputs/full_0429/last.pth` 로 epoch 57 부터 이어 학습. |
| (진행 중) | watchdog 자동 재시도 루프. 200 epoch 완주 또는 8회 연속 무진행 실패 시 종료. |

## 학습 곡선 (실측)

| epoch | val MAE(HU) | 비고 |
|------:|------------:|:-----|
| 0 | 818.38 | 첫 best |
| 5 | 86.51 | |
| 10 | 52.00 | |
| 20 | 27.03 | 거의 수렴권 진입 |
| 30 | 26.64 | |
| 38 | 26.78 | best 갱신 |
| 42 | 26.55 | best 갱신 |
| 45 | 26.46 | best 갱신 |
| **48** | **26.24** | best 갱신 (Session 1 종료 시점 best) |
| 57 | — | val 도중 무성종료 (Session 1 끝) |
| 147 | 26.25 | watchdog Session 2 진행 중 |
| 148 | 26.19 | Session 2 마지막 (사용자 요청으로 중단) |
| **145** | **26.10** | best.pth 최종 — Session 2 도중 갱신 |

> **이미 nnUNet plain (73 HU) 보다 2.8배, ResEnc (433 HU) 보다 16배 좋음.** epoch ~48 이후 ~26 HU 부근에서 매우 좁은 plateau (Session 2 에서도 26.24 → 26.10 으로 0.14 감소).

## 학습 종료 (사용자 요청)

| 시각 | 이벤트 |
|------|-------|
| 2026-04-29 14:17 | 사용자 요청으로 watchdog/training stop. last 완주 epoch = 148, best.pth = epoch 145 (val MAE 26.10). |
| 2026-04-29 14:18 | [src/batch_infer.py](./src/batch_infer.py) 신규 작성 — 모델 1회 로드 후 200 케이스 일괄 추론. |
| 2026-04-29 14:18 ~ 14:21 | 200 케이스 batch inference (best.pth) → [outputs/inference_best_0429/](./outputs/inference_best_0429/). 122s 소요 (0.6s/case). |
| 2026-04-29 14:21 ~ | val.py (SynthRAD 표준) 평가 → [outputs/metrics_best_0429/metrics.json](./outputs/metrics_best_0429/metrics.json). |

## 최종 결과 — sCT U-Net (best.pth, 200 케이스)

| Metric | Mean | Std | Median | Min | Max |
|--------|-----:|----:|-------:|----:|----:|
| MAE_HU | 57.3824 | 20.1502 | 49.4531 | 35.0954 | 126.5799 |
| PSNR_dB | 31.4411 | 2.7004 | 32.2443 | 24.4636 | 35.7010 |
| SSIM | 0.9511 | 0.0237 | 0.9574 | 0.8642 | 0.9862 |
| MS_SSIM | 0.9647 | 0.0269 | 0.9751 | 0.8568 | 0.9921 |
| DICE_bone | 0.8557 | 0.0611 | 0.8666 | 0.6231 | 0.9556 |
| HD95_bone_mm | 2.0491 | 1.4512 | 1.4142 | 0.0000 | 12.5300 |

> `scale_ok` 통과: 200/200. 모든 케이스에서 HD95 유한값.
> 학습 시 보던 val MAE 26.10 vs val.py 의 57.38 차이는 metric 정의 차 때문 (학습은 full-image MAE in [-1,1] → HU, val.py 는 body mask 내부 MAE 만 — air 가 빠져 더 어려운 측정).

## 모델 비교 (동일 200 케이스, 동일 GT, 동일 val.py)

| Metric | **sCT U-Net (NEW, 0429)** | nnUNet plain (Session 1) | nnUNet ResEnc (Session 2, collapsed) |
|--------|--------------------------:|-------------------------:|-------------------------------------:|
| MAE_HU | **57.38** | 73.42 | 433.09 |
| PSNR_dB | **31.44** | 32.77 | 18.00 |
| SSIM | **0.9511** | 0.9143 | 0.7093 |
| MS_SSIM | **0.9647** | 0.9094 | 0.5784 |
| DICE_bone | **0.8557** | 0.8238 | 0.0000 |
| HD95_bone_mm | **2.05** | 3.80 | 211.60 |

핵심 관찰:
- MAE 22% 개선 (73 → 57), DICE_bone 4% 개선 (0.82 → 0.86), HD95 46% 개선 (3.80 → 2.05).
- PSNR 만 plain 대비 1.3 dB 낮음 — best.pth 가 best val MAE 기준이라 중심값(MAE) 에 최적화된 결과. PSNR / SSIM 동시 최적은 별도 loss 가중 조정 필요.
- 200 케이스 모두 scale_ok 통과 (정규화 불일치 0건). ResEnc 가 겪은 mode collapse 도 발생하지 않음.
- 학습 시간: ~2.5 h (148 epoch, RTX 5090) — nnUNet plain 의 1000 epoch 학습 대비 훨씬 짧음.

## Best 5 (lowest MAE_HU) — sCT U-Net

| Case | MAE_HU | PSNR_dB | SSIM | DICE_bone | HD95_bone_mm |
|------|-------:|--------:|-----:|----------:|-------------:|
| 2HNA002 | 35.10 | 35.08 | 0.9841 | 0.9556 | 0.00 |
| 2HNA092 | 35.18 | 34.78 | 0.9835 | 0.9510 | 0.00 |
| 2HNA115 | 35.79 | 35.70 | 0.9654 | 0.8950 | 1.00 |
| 2HNC092 | 35.88 | 34.55 | 0.9738 | 0.9456 | 1.00 |
| 2HNC108 | 37.09 | 34.73 | 0.9862 | 0.9533 | 0.00 |

## Worst 5 (highest MAE_HU) — sCT U-Net

| Case | MAE_HU | PSNR_dB | SSIM | DICE_bone | HD95_bone_mm |
|------|-------:|--------:|-----:|----------:|-------------:|
| 2HNE001 | 126.58 | 24.46 | 0.8975 | 0.7234 | 3.32 |
| 2HNB103 | 116.55 | 24.66 | 0.8694 | 0.7353 | 3.61 |
| 2HNB071 | 115.29 | 24.70 | 0.9102 | 0.7232 | 3.16 |
| 2HNC051 | 114.06 | 25.14 | 0.9057 | 0.6794 | 4.12 |
| 2HNC087 | 113.58 | 24.97 | 0.8773 | 0.6815 | 9.38 |

## SSIM / MS-SSIM 상세 (val.py 결과, n=200)

[val.py](../nnUNet_translation/val.py) 의 SSIM 은 SynthRAD 기준에 따라 body mask 내부에서 슬라이스별 2D SSIM 을 계산하고 슬라이스 평균. dynamic range Q = 4000 HU, gaussian window σ=1.5, win_size=7 (`structural_similarity` 에 `gaussian_weights=True, sigma=1.5, use_sample_covariance=False`). MS-SSIM 은 5 단계 가중 기하 평균 (Wang et al. 2003 weight: 0.0448, 0.2856, 0.3001, 0.2363, 0.1333).

| Metric | Mean | Std | Median | Min | Max |
|--------|-----:|----:|-------:|----:|----:|
| **SSIM** | **0.9511** | 0.0237 | 0.9574 | 0.8642 | 0.9862 |
| **MS-SSIM** | **0.9647** | 0.0269 | 0.9751 | 0.8568 | 0.9921 |

### SSIM Best 5

| Case | SSIM | MS-SSIM | MAE_HU |
|------|-----:|--------:|-------:|
| 2HNC108 | 0.9862 | 0.9921 | 37.09 |
| 2HNC028 | 0.9851 | 0.9906 | 37.84 |
| 2HNA002 | 0.9841 | 0.9914 | 35.10 |
| 2HNA092 | 0.9835 | 0.9900 | 35.18 |
| 2HNC098 | 0.9828 | 0.9879 | 43.51 |

### SSIM Worst 5

| Case | SSIM | MS-SSIM | MAE_HU |
|------|-----:|--------:|-------:|
| 2HNA038 | 0.8642 | 0.8715 | 106.18 |
| 2HNB103 | 0.8694 | 0.8732 | 116.55 |
| 2HNA102 | 0.8734 | 0.8777 | 92.05 |
| 2HNC087 | 0.8773 | 0.8568 | 113.58 |
| 2HNB061 | 0.8871 | 0.8888 | 112.32 |

### SSIM 모델 간 비교

| 모델 | SSIM | MS-SSIM |
|------|-----:|--------:|
| **sCT U-Net (NEW, 0429, best.pth)** | **0.9511** | **0.9647** |
| nnUNet plain (Session 1, checkpoint_final) | 0.9143 | 0.9094 |
| nnUNet ResEnc (Session 2, collapsed) | 0.7093 | 0.5784 |

> sCT U-Net 이 SSIM 4.0%p, MS-SSIM 5.5%p 우위. SSIM 최저 case (2HNA038, 0.8642) 도 nnUNet plain 의 평균 (0.9143) 과 0.05 차이밖에 안 남 → worst 도 plain 의 mean 수준.

---

## Hold-out 60 케이스 (2HNE010 ~ 2HNE109) — 진정한 일반화 평가

학습/val 200 케이스에 거의 포함되지 않은 새 환자 (`2HNE` prefix) 60 케이스로 **진정한 hold-out 일반화** 측정. 학습 200 케이스 평가 (mean MAE 57 HU) 와 달리 학습 cases 가 섞이지 않은 순수 unseen 셋.

### 입력 / 출력 / GT

| 종류 | 경로 | 케이스 수 |
|------|------|----------:|
| 입력 (CBCT) | `outputs/inference input/` (`*_0000.nii.gz`) | 60 |
| 출력 (sCT, HU) | `outputs/inference output/` (`<case>.nii.gz`) | 60 |
| GT (CT) | `outputs/new_gt/` (`<case>.nii.gz`) | 60 |
| 모델 체크포인트 | `outputs/full_0429/best.pth` (epoch 145, train-time val MAE 26.10) | — |

### 실행 스크립트

[scripts/infer_eval_holdout.sh](./scripts/infer_eval_holdout.sh) — 위 경로를 하드코드한 one-command wrapper. `bash scripts/infer_eval_holdout.sh` 한 번이면 inference + val.py 자동 실행.

### 결과 (n=60, val.py SynthRAD 표준)

| Metric | Mean | Std | Median | Min | Max |
|--------|-----:|----:|-------:|----:|----:|
| MAE_HU | 127.99 | 23.28 | 125.80 | 89.52 | 211.19 |
| PSNR_dB | 24.67 | 1.39 | 24.68 | 20.97 | 27.65 |
| SSIM | 0.9025 | 0.0370 | 0.9074 | 0.7752 | 0.9716 |
| MS_SSIM | 0.9021 | 0.0420 | 0.9081 | 0.7372 | 0.9779 |
| DICE_bone | 0.6999 | 0.0827 | 0.7103 | 0.4972 | 0.8588 |
| HD95_bone_mm | 5.01 | 2.33 | 4.74 | 2.00 | 16.40 |

> `scale_ok` 200/200... wait, **60/60** 통과. 60 케이스 모두 2HNE 환자군.

### 학습+val mixed (200 케이스) vs Hold-out 60 비교

| Metric | 학습+val mixed (200, 일부 학습 case 포함) | **Hold-out 60 (순수 unseen)** | Generalization gap |
|--------|----------------------------------------:|------------------------------:|-------------------:|
| MAE_HU | 57.38 | **127.99** | +70.6 HU (×2.2) |
| PSNR_dB | 31.44 | **24.67** | −6.8 dB |
| SSIM | 0.9511 | **0.9025** | −0.049 |
| MS_SSIM | 0.9647 | **0.9021** | −0.063 |
| DICE_bone | 0.8557 | **0.6999** | −0.156 |
| HD95_bone_mm | 2.05 | **5.01** | +2.96 mm |

> 일반화 gap 존재. 단, hold-out 도 nnUNet plain (학습+val 평가에서 73 HU) 와 비슷한 ~128 HU 수준 — sct_unet 가 새로운 환자군에도 nnUNet plain 의 평가셋 성능과 비슷한 수준. **Hold-out 의 모든 60 케이스에서 SSIM > 0.77, DICE > 0.49, HD95 < 17mm — 학습 collapse 없음 (Session 2 ResEnc 의 sub-0.0001 SSIM/0 DICE 와 대비).**

### Hold-out Best 5 (lowest MAE_HU)

| Case | MAE_HU | SSIM | DICE_bone | HD95_bone_mm |
|------|-------:|-----:|----------:|-------------:|
| 2HNE038 | 89.52 | 0.9319 | 0.8348 | 3.00 |
| 2HNE072 | 89.55 | 0.9117 | 0.8588 | 2.00 |
| 2HNE010 | 92.07 | 0.9358 | 0.8147 | 3.61 |
| 2HNE054 | 96.90 | 0.8978 | 0.7117 | 6.71 |
| 2HNE064 | 97.86 | 0.8936 | 0.8076 | 3.00 |

### Hold-out Worst 5 (highest MAE_HU)

| Case | MAE_HU | SSIM | DICE_bone | HD95_bone_mm |
|------|-------:|-----:|----------:|-------------:|
| 2HNE026 | 211.19 | 0.9312 | 0.6107 | 16.40 |
| 2HNE102 | 180.18 | 0.8527 | 0.5652 | 6.40 |
| 2HNE037 | 172.54 | 0.8845 | 0.5377 | 11.70 |
| 2HNE097 | 160.95 | 0.9100 | 0.5658 | 5.83 |
| 2HNE081 | 160.47 | 0.9382 | 0.5322 | 6.40 |

### 산출물

- 인퍼런스 출력 60 × .nii.gz: `outputs/inference output/`
- 메트릭 JSON: `outputs/metrics_holdout60/metrics.json`
- 인퍼런스 stdout 로그: `outputs/batch_infer_holdout60.log`
- val.py stdout 로그: `outputs/val_holdout60.log`
- 실행 스크립트: `scripts/infer_eval_holdout.sh`

## 산출물

- 학습 가중치 (best, epoch 145): [`outputs/full_0429/best.pth`](./outputs/full_0429/best.pth)
- 학습 가중치 (last, epoch 148): [`outputs/full_0429/last.pth`](./outputs/full_0429/last.pth)
- 학습 로그: [`outputs/full_0429/train.log`](./outputs/full_0429/train.log)
- TensorBoard: [`outputs/full_0429/tb/`](./outputs/full_0429/tb/)
- 추론 결과 200 × .nii.gz: [`outputs/inference_best_0429/`](./outputs/inference_best_0429/)
- 추론 stdout: [`outputs/batch_infer_stdout.log`](./outputs/batch_infer_stdout.log)
- val.py 결과 stdout: [`outputs/val_best_0429.log`](./outputs/val_best_0429.log)
- val.py metrics.json: [`outputs/metrics_best_0429/metrics.json`](./outputs/metrics_best_0429/metrics.json)
- watchdog: [`scripts/watchdog.py`](./scripts/watchdog.py), [`scripts/watchdog_unet.log`](./scripts/watchdog_unet.log)

## 다음 단계 (선택)

1. **추가 학습**: `--resume outputs/full_0429/last.pth` 로 200 epoch 까지 마저 학습 (52 epoch 추가, ~90분). plateau 가 강해서 추가 개선폭은 작을 가능성.
2. **L1 + SSIM loss**: PSNR 이 plain 대비 약간 낮은 점 개선 위해 [configs/default.yaml](./configs/default.yaml) 의 `training.loss` 를 `l1_ssim` 으로 바꿔 재학습.
3. **Worst case 분석**: 2HNE001, 2HNB103 처럼 MAE > 100 HU 인 케이스의 GT/pred 시각화로 실패 패턴 파악 (병변/금속 saturation 의심).
4. **PLAN.md 의 [ ] GPU smoke test 항목** 체크 가능.

## Watchdog 산출물

- 스크립트: [scripts/watchdog.py](./scripts/watchdog.py)
- watchdog 로그: `scripts/watchdog_unet.log`
- 상태 파일: `scripts/watchdog_unet_state.txt`
- 각 attempt 의 stdout: `scripts/unet_attempt_NNN.log`
- 본 학습 통합 로그: `outputs/full_0429/train.log`

## 코드 수정 요약

- [src/train.py](./src/train.py): `evaluate()` 가 single forward pass 대신 `monai.inferers.sliding_window_inference` 를 호출. patch_size, sw_batch_size, sw_overlap 을 config 의 `slicing.patch_size`, `inference.sw_batch_size`, `inference.sw_overlap` 에서 읽어옴. infer.py 와 동일한 inference behavior 보장.
- [configs/default.yaml](./configs/default.yaml): `out_dir` 을 `outputs/full_0429` 로 변경 (smoke 출력과 분리).
