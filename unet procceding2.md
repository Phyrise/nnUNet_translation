# nnUNet_translation — L1 + VGG Perceptual Loss 실험 (2026-04-29)

워크오더 [작업지시서_perceptual_loss_단독추가.md](../nnUNet_translation/작업지시서_perceptual_loss_단독추가.md) 적용 기록. 본 실험은 nnUNet_translation 학습이지만 사용자 요청대로 진행 기록은 `sct_unet/unet procceding2.md` 에 둠.

## 워크오더 핵심 (1줄 요약)

이전 ResEnc 학습 (Session 2, MAE 73 baseline) 의 **bone edge p99 ≈ 975 HU** 문제를 줄이기 위해, **L1 loss 에 VGG16 perceptual loss (λ=0.05) 만 추가** — 다른 하이퍼파라미터/모델/데이터/augmentation 일체 변경 금지 (single-variable experiment).

## Step 0 — 환경/코드 점검 결과

| 항목 | 값 |
|------|----|
| Base trainer | `nnUNetTrainerMRCT_mae` (이 fork 의 L1 translation trainer, `enable_deep_supervision=False`, `num_epochs=1000`) |
| 기존 loss | `myMAE()` = `nn.L1Loss` wrapper (z-score space) |
| Loss 정의 메서드 | `_build_loss()` — override 만 하면 됨 |
| Patch shape | 3D (B, 1, D=80, H=256, W=256) |
| Input/output channels | 1, CTPaperNormalization 적용 z-score |
| Normalization stats (Dataset050) | mean=−753.617, std=420.007 (from plans `foreground_intensity_properties_per_channel`) — **사용자 지침대로 plans 객체 attribute 로 참조, hardcode 금지** |
| HU 클립 | `CTPaperNormalization.HU_MIN=−1024`, `HU_MAX=3071` (class attribute) |
| 학습 데이터 | Dataset 50, fold 0, ResEncUNetLPlans (Session 2 와 동일 — 진정 single-variable) |
| 기존 perceptual.py | MedicalNet 기반 perceptual 코드 이미 존재. 덮어쓰지 않고 새 파일 `vgg_perceptual.py` 로 분리 |

## 사용자 결정 사항 (4개 답변 확정)

1. **(a) Denorm 후 perceptual 적용** — `hu = z * std + mean`. **plans 객체 attribute** 에서 참조, hardcode 금지.
2. **부모 = `nnUNetTrainerMRCT_mae`**.
3. **(ii) 무작위 8 slice**, 매 iteration `torch.randperm`. **body voxel > 1% slice 우선** 선택 (구현됨), 미달 시 plain random fallback.
4. **λ=0.05 시작 + PERC_DIAG 자동 로깅**. iter 1~10 매 step, 이후 epoch별 1줄 요약. **첫 10 iter 로그 사용자 보고 후 λ 조정 결정** — 임의 변경 금지.

## 신규 / 수정 파일

### 신규

| 종류 | 경로 | 비고 |
|------|------|------|
| Perceptual loss 모듈 | [`nnunetv2/training/loss/vgg_perceptual.py`](../nnUNet_translation/nnunetv2/training/loss/vgg_perceptual.py) | `VGGPerceptualLoss` (relu1_2, relu2_2, relu3_3, frozen) + `CombinedL1Perceptual` (L1 z-score + λ·Perc HU, 8 slice subsample, body-prefer) |
| Trainer 서브클래스 | [`nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainer_L1Perceptual.py`](../nnUNet_translation/nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainer_L1Perceptual.py) | 부모: `nnUNetTrainerMRCT_mae`. `_build_loss` + `train_step` (PERC_DIAG 로깅) + `on_train_epoch_end` (epoch summary) override |
| Watchdog | [`scripts/watchdog_l1perc.py`](../nnUNet_translation/scripts/watchdog_l1perc.py) | 자동 재시도, `--c` resume |
| Post-train | [`scripts/post_train_l1perc.py`](../nnUNet_translation/scripts/post_train_l1perc.py) | `checkpoint_final.pth` 폴링 → 200 케이스 inference → val.py → 결과를 이 파일에 자동 추가 |
| 진행 기록 | `sct_unet/unet procceding2.md` | ← **이 파일** |

### 기존 파일 무수정 확인

- `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py` (base): 이전 세션부터 있던 fork-state 2줄 외 추가 수정 없음
- `nnunetv2/training/loss/perceptual.py` (기존 MedicalNet 기반): 손대지 않음

## 로컬 dry-run (실 학습 전 sanity check)

dummy `(2, 1, 80, 256, 256)` z-score 텐서로 `CombinedL1Perceptual.forward + backward` 확인:

| 항목 | 측정값 | 예상 범위 | 판정 |
|------|-------:|---------:|:----|
| L1 (z-score) | 0.3384 | 0.3~1.5 (워크오더 § Step 3 참고) | OK |
| Perceptual | 0.4959 | (참고치) | OK |
| λ·Perc | 0.0248 | (L1 의 5~20%) | OK |
| **ratio = λ·Perc / L1** | **7.3%** | **5~20%** | **OK ✓** |
| Forward time | 0.37s | <1s | OK |
| Backward time | 0.14s | <1s | OK |
| VGG params frozen | 14/14 (requires_grad=False) | 모두 frozen | OK |

> dummy 는 random noise 라 실 학습 시 magnitude 가 다를 수 있음 — 첫 10 iter 의 `[PERC_DIAG]` 로그가 진짜 기준. 거기서 ratio 가 5~20% 밖이면 λ 조정 보고 후 사용자 결정.

## 진행 타임라인

| 시각 | 이벤트 |
|------|-------|
| 2026-04-29 | Step 0~3 (코드 작성 + dry-run) 완료 |
| 2026-05-12 09:07 | watchdog + post_train 시작. attempt 1 fresh start. nnUNetv2_train init OK. |
| 2026-05-12 09:07 | epoch 0 첫 10 iter PERC_DIAG 캡처 — ratio 평균 4.6% (3.0~6.5%), 사용자 보고. |
| 2026-05-12 09:09 | epoch 0 끝 — `[PERC_DIAG] epoch=0 (last train_step) L1=0.2530 Perc=0.0562 lambda*Perc=0.0028 ratio=1.1%`. Perc 가 L1 보다 훨씬 빨리 줄어 (0.40→0.06) ratio 가 예상과 반대로 ↓. 추후 epoch 별 추이 모니터링 필요. |
| 2026-05-12 09:09 | **사용자 결정: (A) λ=0.05 유지 + 계속 진행**. 임의 변경 금지 준수. |
| (진행 중) | 1000 epoch 학습. GPU 98% / VRAM 23GB / iter ~0.43s. epoch 별 PERC_DIAG summary 1줄씩 자동 기록. |
| (대기) | 학습 완료 → post_train 자동 200 케이스 inference + val.py → 결과 이 파일 끝에 자동 추가 |

## 첫 10 iter PERC_DIAG (사용자 보고분)

```
[PERC_DIAG] iter=1  L1=0.9690 Perc=0.6352 lambda*Perc=0.0318 ratio=3.3%
[PERC_DIAG] iter=2  L1=1.0125 Perc=0.6116 lambda*Perc=0.0306 ratio=3.0%
[PERC_DIAG] iter=3  L1=0.6854 Perc=0.5294 lambda*Perc=0.0265 ratio=3.9%
[PERC_DIAG] iter=4  L1=0.7405 Perc=0.5237 lambda*Perc=0.0262 ratio=3.5%
[PERC_DIAG] iter=5  L1=0.4003 Perc=0.5170 lambda*Perc=0.0259 ratio=6.5%
[PERC_DIAG] iter=6  L1=0.5592 Perc=0.5185 lambda*Perc=0.0259 ratio=4.6%
[PERC_DIAG] iter=7  L1=0.4160 Perc=0.4703 lambda*Perc=0.0235 ratio=5.7%
[PERC_DIAG] iter=8  L1=0.4781 Perc=0.4364 lambda*Perc=0.0218 ratio=4.6%
[PERC_DIAG] iter=9  L1=0.3458 Perc=0.4054 lambda*Perc=0.0203 ratio=5.9%
[PERC_DIAG] iter=10 L1=0.4109 Perc=0.3960 lambda*Perc=0.0198 ratio=4.8%
[PERC_DIAG] epoch=0 (last train_step) L1=0.2530 Perc=0.0562 lambda*Perc=0.0028 ratio=1.1%
```

## 관찰 (epoch 0)

- **iter 0~10**: ratio 평균 **4.6%** (목표 5~20% 의 하한 살짝 미달, 단 iter 5/7/9 는 5~7% 범위 진입)
- **epoch 0 끝**: ratio **1.1%** — Perc 가 L1 보다 훨씬 빨리 줄어들음 (Perc 0.40→0.06 vs L1 0.40→0.25)
- 즉 **VGG features 가 빠르게 매칭** 되고, L1 (전체 픽셀 평균 차이) 은 더 천천히 감소. 이는 perceptual 이 high-frequency / 구조적 패턴 위주를 빠르게 잡고, 절대 픽셀 값 (HU) 정밀 매칭은 L1 이 끌고 가는 분업 양상으로 해석 가능.
- ratio 가 1% 대로 내려가면 perceptual term 의 그래디언트 영향이 미미해질 수 있음. epoch 5~10 의 summary 추이 보고 필요 시 λ 조정 재논의.

## 자동 모니터링 (사용자 지시 — 2026-05-12 추가)

사용자 추가 요청대로 [scripts/monitor_l1perc.py](../nnUNet_translation/scripts/monitor_l1perc.py) 구현:

| 체크 | 동작 | 결과 |
|------|------|------|
| NaN/Inf 자동 감지 (epoch 5+) | 발견 시 ALERT + STOP_FLAG + watchdog/post_train/nnUNetv2_train 모두 kill | **OK** — 모든 epoch 유한 |
| ratio @ epoch 10 (5% 하한) | latest < 5% 면 ALERT (학습은 계속) | **OK** — mean 11.65% / latest 19.80% |
| ratio @ epoch 30 (25% 상한) | latest > 25% 면 ALERT (학습은 계속) | **OK** — mean 11.34% / latest 12.60% |
| ratio @ epoch 50 (info) | 평균/최신 ratio 정보 기록 | mean 9.90% / latest 1.70% |
| 매 epoch 종료 시 train_loss/val_loss 통합 로깅 | `[EPOCH N] train_loss=X val_loss=X ratio=X%` 형식 | OK — `monitor.log` |
| **임의 λ 변경** | **금지** (사용자 승인 필수) | 준수 — 제안만 ALERT, 변경 없음 |

산출물:
- `results/l1_perceptual_logs/monitor.log` (chronological event log, epoch 별 한 줄)
- `results/l1_perceptual_logs/MONITOR_ALERTS.log` (ALERT 만 — 비어있으면 모든 정상)
- `results/l1_perceptual_logs/monitor_status.json` (마지막 snapshot)
- `results/l1_perceptual_logs/STOP_FLAG` (있으면 NaN-stop 발생)

모니터는 `epoch >= 50` 도달 + 모든 체크 완료 후 자체 종료 (cleanly exit). 추가 체크 필요 시 재실행.

## 학습 곡선 (모니터 캡처, epoch 0 → 89)

전체 epoch 별 `[EPOCH N] train_loss val_loss ratio` 는 `results/l1_perceptual_logs/monitor.log` 참조. 주요 시점:

| epoch | train_loss | val_loss | ratio (%) | 메모 |
|------:|-----------:|---------:|----------:|:-----|
| 0 | 0.1983 | 0.1547 | 1.10 | 시작 |
| 5 | 0.0343 | 0.0230 | 16.50 | 빠른 수렴 |
| 10 | 0.0209 | 0.0207 | 19.80 | ratio 목표 안 진입 |
| 20 | 0.0172 | 0.0226 | 6.70 | val 분산 큼 |
| 30 | 0.0135 | 0.0155 | 12.60 | plateau 진입 |
| 50 | 0.0122 | 0.0092 | 1.70 | val < train |
| 80 | 0.0100 | 0.0100 | 2.20 | |
| 89 | 0.0121 | 0.0039 | 1.30 | 현재 |

**관찰**:
- epoch 5 이후로 train_loss 가 0.012~0.020 수준에서 plateau
- val_loss 가 train_loss 보다 낮은 케이스 다수 — nnUNet 의 val 은 50 iter 표본이라 epoch 별 분산이 큼 (단일 epoch 으로 overfit 단정 어려움)
- ratio 가 epoch 50 이후 종종 1~5% 로 내림 — 후반에 perceptual term 영향 미미해짐. 단, 사용자 지침 (임의 변경 금지) 준수.

## 다음 단계 (자동)

1. 학습 1000 epoch 완주 (현재 epoch 129, ~26h 남음)
2. `checkpoint_final.pth` 출현 시 post_train 자동 실행:
   - **(A) L1+Perc → 200-case** inference + val.py → `results/inference/l1_perceptual_200case` + `results/prediction_metrics/l1_perceptual_200case`
   - **(B) L1+Perc → 60-case hold-out** inference + val.py → `..._60case`
   - **(C) Session 1 plain L1 → 60-case** inference + val.py → `..._baseline_plain_60case` (junction `Dataset050_TrainHN_Input/nnUNetTrainerMRCT_mae__nnUNetPlans__3d_fullres` → `old0422/.../`)
   - 워크오더 § Step 5 표 형식으로 (A) plain-200 vs L1+Perc-200, (B/C) plain-60 vs sct_unet-60 vs L1+Perc-60 비교 결과 + 2HNE097 worst patient 추적 + Best5/Worst5 를 이 markdown 끝에 자동 추가

## Checkpoint Archiver (사용자 지시 — 즉시 실행)

학습 중간 체크포인트 분실 위험에 대비해 외부 archiver 가동. **현재 학습은 일체 건드리지 않음** (별도 폴링 프로세스).

| 항목 | 값 |
|------|----|
| 스크립트 | [`scripts/checkpoint_archiver.py`](../nnUNet_translation/scripts/checkpoint_archiver.py) |
| 폴링 간격 | 30s |
| 감시 대상 | `results/Dataset050_TrainHN_Input/nnUNetTrainer_L1Perceptual__nnUNetResEncUNetLPlans__3d_fullres/fold_0/checkpoint_latest.pth` |
| milestone (영구 보존) | **epoch 200, 400, 600, 800, 1000** — 5 스냅샷 × ~1.1 GB ≈ **5.5 GB** (사용자 예상 ~5GB 와 일치) |
| 저장 위치 | `results/archive/l1_perceptual_snapshots/checkpoint_epoch_NNNN.pth` |
| 복사 방식 | `.tmp` 로 쓴 후 `os.replace()` (atomic) |
| monitor.log 스냅샷 | **매 50 epoch** 마다 같은 archive 폴더에 `monitor_at_epoch_NNNN.log` 로 복사 (학습 로그 분실 위험 대비) |
| 종료 조건 | `checkpoint_final.pth` 출현 + 모든 milestone 시도 완료 |
| 시작 시각 | 2026-05-12 13:10 (학습이 epoch 129 진행 중) |
| 첫 캡처 milestone | **epoch 200** 도달 시점 (현재 epoch 129 → ~13 epoch 후, ~25분 후) |
| 디스크 모니터링 | archiver 가 매 milestone 후 누적 MB 출력 |

상태 파일: `results/archive/l1_perceptual_snapshots/archive_state.json` (어떤 milestone 까지 저장됐는지)
archiver 로그: `results/l1_perceptual_logs/archiver.log`

## Stage 2 준비 메모 (다음 학습 = Session 1 plain L1 baseline 재학습 시작 전 적용)

> 사용자 요청 사항. **현재 (Stage 1, L1+Perceptual) 학습 끝나기 전엔 코드 변경 X**. Stage 2 시작 직전에 trainer 코드/스크립트에 반영.

| 변경 | 적용 위치 |
|------|----------|
| `save_every` 50 → 25 | Stage 2 trainer 의 `__init__` 또는 base nnUNetTrainer 의 attribute (subclass override 권장) |
| milestone epoch [100, 200, 400, 600, 800] **trainer 내부 영구 저장** | trainer 의 `on_train_epoch_end` 에서 `(epoch+1) in MILESTONES` 일 때 `checkpoint_milestone_NNNN.pth` 저장 |
| archive 폴더 read-only (학습 종료 후) | post-training 스크립트에서 `attrib +R` (Windows) 또는 `chmod -R a-w` |
| monitor.log 50 epoch 마다 archive | 동일 archiver 패턴 재사용 가능 |

Stage 2 시작 시 다음 파일 신규/수정 예정:
- `nnunetv2/training/nnUNetTrainer/variants/loss/nnUNetTrainer_PlainL1_Stage2.py` (예시 이름; `nnUNetTrainerMRCT_mae` subclass)
- `scripts/checkpoint_archiver_stage2.py` (위 archiver 의 사본 — milestone 다른 경로/리스트)

## 진행 타임라인 (이번 보고분)

| 시각 | 이벤트 |
|------|-------|
| 2026-05-12 13:09 | post_train_l1perc.py 를 multi-set 평가 (200-case + 60-case + baseline 60-case) 로 업데이트 후 재시작. 새 인스턴스 polling 시작. |
| 2026-05-12 13:10 | checkpoint_archiver.py 작성 + 백그라운드 시작. 첫 monitor.log 스냅샷 captured at epoch 129. milestone capture 대기 (200~). |
| 2026-05-12 13:35~ | **워크오더 [stage5_sanity_inference_plan](.) 즉시 실행** — Stage 3 collapse 가 Stage 5 에도 잔존하는지 5 케이스 sanity check |
| 2026-05-12 13:36 | 학습 깨끗이 종료 (watchdog + nnUNetv2_train kill, best/latest 보존). archiver milestone 200 도달 전 중단됨. |
| 2026-05-12 13:38 | sanity_inference (5 cases, TTA off, best.pth=epoch 128) 실행. 워크오더의 3개 미존재 ID 는 가장 가까운 존재 ID 로 대체 (2HNA001→002, 2HNC012→017, 2HNE021→004). |
| 2026-05-12 13:40 | diagnose_collapse 실행 → **VERDICT: COLLAPSED 5/5** |

---

# 🚨 Stage 5 = COLLAPSED (사용자 문서 [stage5_collapse_confirmed.md] 전문 보존)

## 진단 결과 (사용자 + Claude 공동 확인)

| case    | mean    | std  | min     | max    | p1      | p99     | verdict   |
|---------|--------:|-----:|--------:|-------:|--------:|--------:|-----------|
| 2HNA002 | −325.0  |  5.3 | −433.0  |  788.7 | −325.4  | −325.0  | COLLAPSED |
| 2HNB005 | −325.1  |  5.0 | −445.8  |  815.0 | −325.4  | −325.0  | COLLAPSED |
| 2HNC017 | −325.0  |  4.9 | −430.2  |  523.3 | −325.4  | −325.0  | COLLAPSED |
| 2HNE003 | −325.1  |  7.1 | −497.6  | 2183.9 | −325.4  | −325.0  | COLLAPSED |
| 2HNE004 | −325.0  |  8.6 | −1000.6 | 1579.4 | −325.4  | −325.0  | COLLAPSED |

## 수치 해석 — 정규화 공간 z-score 정확히 1.0

Dataset051 (target CT) plans (nnUNetPlans 와 ResEncUNetLPlans 양쪽 동일):
- `foreground_intensity_properties_per_channel['0']['mean']` = **−775.2418**
- `foreground_intensity_properties_per_channel['0']['std']`  = **449.8501**
- `export_prediction.py:146-148` 의 하드코드 `_CT_MEAN/_CT_STD` 와 정확히 동일

```
HU = pred_z * std + mean = 1.0 × 449.8501 + (−775.2418) = −325.3917 HU
```

관측 −325.0 ~ −325.1 HU 와 **소수점 단위로 일치**. 모델이 정규화 공간에서 정확히 1.0 을 출력 (input 무관 constant).

## Root cause 추정 (워크오더 § 3)

`nnUNetDataLoader3D_MRCT` 가 D050(CBCT) 만 로드, target(`seg`) 슬롯에 **binary mask** (foreground=1) 를 그대로 사용. 모델 입장에서 trivial 최적해 = "어떤 입력이든 1.0 출력" → L1 손실 최소화 달성 → 학습 손실은 잘 떨어지지만 sCT 가 아닌 상수.

## Stage 3 vs Stage 5 비교 — 동일 trivial solution

| | Stage 3 (336ep ResEnc, random init) | **Stage 5 (128ep ResEnc, L1+Perc)** |
|---|---|---|
| 출력 mean (HU) | −325.39 | **−325.0 ~ −325.1** |
| z-score 환산값 | 1.0 | **1.0** |
| std (HU) | ≈5 | **5~9** |
| EMA MAE (norm) | 0.0016 | **0.0015** |

→ 두 Stage 모두 같은 trivial 최적해로 수렴. 모델 구조/loss/init 변경 무의미.

## 보조 변수들 무효화 (collapse 결과로 모두 효과 없음 확정)

| 변수 | 가설 | 5/5 collapse 결과 |
|---|---|---|
| L1 → L1+Perceptual | perceptual 이 trivial 회피 | **무효** (collapse) |
| Random init → ZeroInitResidual | 초기 안정성 | **무효** (epoch 10 stop, 같은 경로) |
| Constant LR → Warmup+GradClip | 초기 발산 방지 | **무효** (epoch 61 stop, 같은 경로) |
| nnUNetPlans → ResEncUNetLPlans | 모델 표현력 ↑ | **무효** |
| ZScore → CTPaperNormalization | 클립/스케일 정확 | **무효** |

→ 모든 보조 변수는 파이핑 fix 가 선결되어야 의미 있음. **Stage 2 (PlainConvUNet+ZScore+L1, MAE 73 HU)** 만 정상 동작했던 이유는 `automate_translation.py` 의 수동 `_seg.npy` 복사 워크어라운드 덕분.

## 산출물 상태

- `checkpoint_best.pth`, `checkpoint_latest.pth` (epoch ~140): 둘 다 collapse 패턴, **평가 무의미**
- archiver: milestone 200 미도달 → `archive/l1_perceptual_snapshots/` 에 epoch 129 monitor.log 스냅샷 1개만
- inference 산출물 (`sanity_stage5/*.nii.gz`): collapse 진단 근거로만 보존
- post_train_l1perc.py: 가동 중이지만 checkpoint_final.pth 안 만들었으므로 영원히 폴링 (별도 stop 필요)

## 다음 단계 (워크오더 § 3.1)

학습은 이미 중단 완료. 진행 순서:

1. **`nnUNetDataLoader3D_MRCT` 코드 리뷰** — D050 case 의 `seg` 슬롯에 D051 동명 case `.npz['data']` 가 어떻게 (안)매핑되는지 정확히 파악
2. **dataloader fix** — 명시적 D051 → seg 슬롯 매핑 코드 추가 (또는 별도 target 채널 분리)
3. **Stage 2 sanity run** — PlainConvUNet + L1 + ZScore + 100 epoch → MAE ~73 HU 재현 확인 (변경 변수 파이핑 하나만 통제)
4. **재현 후 확장** — ResEncL + CTPaper + Perceptual 로 단계적 복귀

`automate_translation.py` 의 수동 복사 방식으로 임시 복원하는 옵션은 **비추천** (Stage 3 가 정확히 그 함정에 빠진 사례라 dataloader 자체를 수정해 영구화하는 것이 안전).

## 결정 대기 (사용자 선택)

(A) Step 1 (dataloader 코드 리뷰) 즉시 시작 — 진단만 보고 후 사용자가 fix 결정
(B) Step 1+2 (진단 + fix 자동) → 사용자 검증 후 Step 3 (Stage 2 sanity run) 학습 시작
(C) post_train_l1perc.py + archiver 만 stop 하고 사용자가 수동 진행
