# Tier 1 E1 학습 진행 상황 (Live snapshot)

작성: 2026-06-02 13:25 / 학습 진행 중 (ep 425/1000, 42.5%). 학습 완료 후 추가 commit 권장.

## 1. 학습 setup [S]

| 항목 | 값 |
|---|---|
| trainer | `nnUNetTrainerMRCT_mae_HUWeight` (신규, 본 commit 산출) |
| dataloader | `nnUNetDataLoader3D_MRCT_PairedFix` (C1 근본 fix, 본 commit 산출) |
| loss | `WeightedL1Loss` (HU range weight: soft=1.5, bone=0.5, air=0.5) |
| 학습 명령 | `nnUNetv2_train 50 3d_fullres 0 -tr nnUNetTrainerMRCT_mae_HUWeight` |
| dataset | D050 (TrainHN CBCT) + D051 (TrainHN CT) 자동 페어링 |
| batch_size | 2 (nnUNetPlans 3d_fullres) |
| patch_size | [56, 192, 192] |
| optimizer | SGD lr=1e-2, momentum=0.99, nesterov, weight_decay=3e-5 (nnUNet default) |
| scheduler | PolyLRScheduler exponent=0.9 |
| epoch | 1000 |
| AMP | True |

근거 파일: [`debug.json`](tier1_e1_snapshot/debug.json) (학습 시작 시 nnUNet 자동 저장).

## 2. 진행 상황 [M]

| 항목 | 값 (13:25 기준) |
|---|---|
| 시작 | 09:57:22 (3h 28m 전) |
| 현재 epoch | **425/1000 (42.5%)** |
| 평균 epoch_time | **~28.3s** |
| 남은 epoch | 575 × 28.3s ≈ 4h 31m |
| **ETA 완료** | **~17:56** (오늘 저녁) |
| GPU | RTX 5090, 8.4GB used / util 97% / 69°C |
| NaN | 없음 |

## 3. epoch trend (최근 6 epoch, training_log 인용)

| ep | train_loss | val_loss |
|---:|---:|---:|
| 397 | 0.0697 | 0.0855 |
| 398 | 0.0811 | 0.1067 |
| 399 | 0.0939 | 0.0937 |
| 400 | 0.0802 | 0.0922 |
| 401 | 0.0775 | 0.0866 |
| 402 | 0.0802 | (진행 중) |

- val_loss oscillation 0.0855~0.1067 (CosineAnnealing 후반 lr 0.00630, 정상 안정 수렴)
- best.pth 갱신 정체 (12:57 = ep ~350 이후) → plateau 진입 신호
- 후반 575 epoch 에서 best 추가 갱신 가능성 [E]

(시각화: [`tier1_e1_snapshot/progress_ep425.png`](tier1_e1_snapshot/progress_ep425.png))

## 4. 본 commit 의 핵심 코드 변경

### 4.1 C1 dataloader fix (근본 수정)

**Bug**: `nnUNetDataLoader3D_MRCT` (기존) 가 D050 만 load, D051 (CT target) 은 별도 로드 메커니즘 부재 → `automate_translation.py` 의 `shutil.copy(D051/*.npy → D050/*_seg.npy)` workaround 의존 → 사이클 간 *_seg.npy 복구 시 collapse 위험 (Stage 5 실증).

**Fix**: `nnUNetDataLoader3D_MRCT_PairedFix` 신규 클래스 (`data_loader_3d.py` 에 추가):
- `__init__` 에 `target_data: nnUNetDataset` (D051) 인자 추가
- case_id 일치 검증 (D050 ⊆ D051, missing 시 RuntimeError)
- `generate_train_batch` 에서 D051 의 같은 case_id 의 data 를 seg 자리에 직접 사용
- workaround 의존성 제거

### 4.2 Tier 1 E1 (HU range weighted L1)

**목적**: phase2 P_hu_range 의 [M] 결과 (MAE_whole 123.19, 4 모델 중 최우수) 를 nnU-Net framework 측에서 재현.

**구현**:
- `WeightedL1Loss` (신규, `loss/weighted_mae.py`): target 을 z-score → HU 역변환 후 HU range 별 weight 적용 (soft [-100,100] = 1.5, bone ≥300 = 0.5, air ≤-800 = 0.5, other = 1.0)
- `nnUNetTrainerMRCT_mae_HUWeight` (신규): MRCT_mae_PairedFix 상속 + `_build_loss` override

**단위 검증 PASS**:
- 모든 diff=0.1 → weighted avg = 0.1 (정상)
- soft 만 error=1.0 → 1.5/4.0 = **0.375** (이론값 정확 일치)

## 5. 검증 plan (학습 완료 후 자동 chain)

| 단계 | 작업 | ETA |
|---|---|---|
| (a) 학습 완료 | 1000 epoch + checkpoint_final.pth + best.pth | ~17:56 |
| (b) phase2 14 case CPU inference | sct_unet/scripts/eval_phase2_test14.py 의 패턴 응용 | ~10분 |
| (c) phase2 evaluate_volume 으로 MAE_whole/soft/bone, SSIM, PSNR 측정 | 13 case fair subset + 14 case caveat | ~5분 |
| (d) 비교 보고서 | section17 신규 — PairedFix+HUWeight vs sct_unet+C4 vs phase2 4 모델 | ~10분 |

## 6. 본 commit 의 산출 파일 (4 신규 + 1 수정)

| # | 파일 | 종류 |
|---|------|------|
| 1 | `nnunetv2/training/dataloading/data_loader_3d.py` | **M** — `nnUNetDataLoader3D_MRCT_PairedFix` 클래스 추가 (+72줄) |
| 2 | `nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainerMRCT_mae_PairedFix.py` | **A** — 신규 trainer (~95줄) |
| 3 | `nnunetv2/training/nnUNetTrainer/variants/network_architecture/nnUNetTrainerMRCT_mae_HUWeight.py` | **A** — 신규 trainer (~30줄) |
| 4 | `nnunetv2/training/loss/weighted_mae.py` | **A** — `WeightedL1Loss` 클래스 (~95줄) |
| 5 | `docs/tier1_e1_status.md` (본 파일) | **A** — 학습 진행 보고 |
| 6 | `docs/tier1_e1_snapshot/{debug.json, progress_ep425.png, training_log_tail.txt}` | **A** — 13:25 시점 스냅샷 |

## 7. 비고

- 학습 완료 (~17:56) 후 본 docs/ 갱신 commit 권장
- results/ 디렉토리 (.gitignore) — checkpoint .pth (245MB) 는 git 외부 보관. GitHub release 또는 별도 storage (HuggingFace, S3 등) 권장
- 관련 보고서: `phase2_unet_weighted/amed plan unet0601.md` (git 외부), `nnUNet_translation/results/section17_C1_dataloader_root_fix.md` (git 외부, ignored)
