# sCT U-Net (MONAI 2.5D) — 재학습 진행 기록 (full_v2)

이전 학습 [unet procceding.md](./unet%20procceding.md) (full_0429, 148 epoch stop) 의 후속 — 200 epoch 완주 + 실시간 progress.png + 자동 inference + val 파이프라인.

## 핵심 결정

| 항목 | 값 | 비고 |
|------|----|------|
| 데이터 (raw NIfTI) | `C:/Users/USER/Desktop/dev/nnUNet_translation/raw/Dataset050_TrainHN_Input/imagesTr` (CBCT) + `Dataset051_TrainHN_Target/imagesTr` (CT) | 그대로 사용 |
| 별도 전처리 | **불필요** — sct_unet 의 [src/data.py](./src/data.py) 가 raw NIfTI 를 직접 읽어 on-the-fly 정규화 (CT [-1000,2000]→[-1,1], CBCT per-volume z-score) |
| Split | nnUNet `splits_final.json` fold 0 (train 160 / val 40) | full_0429 와 동일 — 동일 평가 조건 |
| 출력 폴더 | `outputs/full_v2/` | full_0429 와 분리 (이전 best.pth 보존) |
| Epochs | 200 | full_0429 와 동일 |
| Loss / lr / model | L1, Adam(1e-4), Cosine, MONAI UNet (32-512) | full_0429 와 동일 |
| 진행 기록 | `unet procceding1.md` (← **이 파일**) | full_0429 의 markdown 과 분리 |

> 사용자 질문 "전처리 다시 해야 할까?": **아니요.** nnUNet 전처리는 nnUNet 만 사용. sct_unet 은 raw NIfTI 직접 읽으므로 추가 전처리 단계 없음.

## 신규 / 수정 파일

| 종류 | 경로 | 비고 |
|------|------|------|
| 학습 스크립트 패치 | [src/train.py](./src/train.py) | 매 epoch 마다 `outputs/full_v2/progress.png` 생성 (3-panel: loss / epoch_time / lr — nnUNet progress.png 와 동일 스타일) + `history.json` 저장. resume 시 history 도 이어받음. |
| 새 config | [configs/full_v2.yaml](./configs/full_v2.yaml) | `out_dir: outputs/full_v2` 만 변경 |
| Watchdog | [scripts/watchdog_sct.py](./scripts/watchdog_sct.py) | crash 시 30s 후 `--resume outputs/full_v2/last.pth` 로 자동 재시작. `Training done.` 발견 시 자체 종료. 최대 50 attempts. |
| Post-train | [scripts/post_train_sct.py](./scripts/post_train_sct.py) | 60s 간격으로 train.log 폴링 → `Training done.` 발견 시 batch_infer 60 케이스 → val.py → **이 markdown 끝에 결과 자동 추가**. v1 (full_0429) 와 직접 비교표 포함. |

## 실시간 progress.png

학습 중 매 epoch 종료 후 [outputs/full_v2/progress.png](./outputs/full_v2/progress.png) 가 갱신됩니다 (atomic rename). VS Code 등 IDE 에서 해당 파일을 열어두면 자동 reload (또는 새로고침) 으로 학습 곡선을 실시간 확인 가능.

3-panel 구성 (첨부 이미지와 동일):
1. **loss** — train loss (blue) + val loss in normalized space (red, MAE_HU / Q 환산)
2. **epoch duration** — 초 단위
3. **learning rate** — Cosine 스케줄

추가로 [outputs/full_v2/history.json](./outputs/full_v2/history.json) 에 epoch / train_loss / val_loss / epoch_time / lr 가 누적 저장됩니다.

## 자동 실행 흐름

```
[Watchdog] watchdog_sct.py
   └─ python -m src.train --config configs/full_v2.yaml [--resume last.pth]
        ├─ 매 epoch: progress.png 갱신, history.json 저장
        └─ "Training done." 출력 시 종료
[Post-train] post_train_sct.py
   ├─ train.log 폴링 (60s)
   ├─ "Training done." 발견 시 batch_infer 60 hold-out
   ├─ val.py 평가
   └─ unet procceding1.md 끝에 결과 섹션 추가
```

둘 다 백그라운드로 실행되어 사용자 개입 0.

## 진행 타임라인

| 시각 | 이벤트 |
|------|-------|
| 2026-04-29 | train.py progress.png/history.json 패치, configs/full_v2.yaml + watchdog_sct.py + post_train_sct.py 작성 완료 |
| 2026-04-29 17:38 | watchdog + post_train 백그라운드 시작. Training 정상 로딩 (160 train / 40 val / 13634 train slices / 6.50M params). |
| 2026-04-29 17:39 | epoch 0 시작. 매 epoch 종료 후 progress.png + history.json 생성 확인. |
| 2026-04-29 17:48 | epoch 4 완료 (avg train loss 0.0797, ~65s/epoch). best/last.pth + progress.png + history.json 모두 정상 갱신 중. |
| 2026-04-29 17:48 | post_train 의 출력 경로를 사용자 요청대로 `outputs/inference output/` (이전 v1 .nii.gz 덮어쓰기) 로 수정 후 재시작. |
| 2026-04-29 17:50 | epoch 5 완료 (val MAE 86.04 HU). |
| 2026-04-29 17:53 | **사용자 결정에 따라 학습 중단 → num_epochs 200 → 1000 으로 변경 → fresh restart**. CosineAnnealingLR 의 T_max 가 1000 으로 새로 만들어져 lr 스케줄 일관성 유지. `outputs/full_v2/` 의 best/last/history/progress 모두 삭제 후 재시작. |
| 2026-04-29 17:53 | 새 watchdog + post_train 시작 (`num_epochs=1000`, `NUM_EPOCHS=1000` 양쪽 일치 확인). |
| 2026-04-29 18:00 | **사용자 요청으로 50 epoch 마다 별도 체크포인트 저장 기능 추가**. epoch 3 까지 진행된 상태에서 stop. |
| 2026-04-29 18:01 | train.py 패치 (`save_every` 옵션 추가) + config 에 `save_every: 50` 추가. last.pth 보존되어 watchdog 가 자동 `--resume` 으로 epoch 4 부터 재개. |
| 2026-04-29 18:01 ~ 19:08 | epoch 4 → 40 정상 진행. val MAE 86 → 26.37 까지 빠르게 수렴. |
| 2026-04-29 19:08 | **사용자 요청으로 학습 중단**. watchdog + post_train + python 모두 종료. GPU idle. |

## 중단 시점 상태 (2026-04-29 19:08)

- **마지막 완료 epoch**: 40 (epoch 0 → 40, 약 75분 걸림 — 그중 18:01~19:08 의 37 epoch 분이 이번 세션)
- **best val MAE 진화**: 86 → 73 → ... → 27.11 → 27.04 → 26.55 → 26.47 → **26.37** (epoch 33, 18:53 시점)
- **최근 val MAE 추이** (epoch 33 ~ 40): 26.44, 26.76, 26.72, 26.68, 26.70, 26.59, 26.70, 26.53 — plateau 진입
- **저장된 가중치**:
  - `checkpoint_best.pth` (18:53, epoch 33, val MAE **26.37**)
  - `checkpoint_latest.pth` (19:08, epoch 40)
  - `checkpoint_epoch_NNNN.pth` 파일 **없음** (첫 snapshot 은 epoch 50 도달 시 — 9 epoch 차이로 미달)

## v1 (full_0429, 200ep run, epoch 145 best 26.10) 와의 비교

| 항목 | v1 (full_0429) | v2 (full_v2, 중단 시) |
|------|---------------:|----------------------:|
| 학습 epoch 도달 | 148 | 40 |
| best val MAE (HU, 학습+val mixed 기준) | 26.10 | 26.37 |
| 학습 시간 | ~2.5h | ~75분 |
| 차이 | — | v1 이 3.7배 더 학습, val MAE 0.27 더 좋음. v2 epoch 40 시점이 v1 의 epoch ~30 수준에 해당. |

> 이번 학습은 v1 과 동일 데이터/하이퍼파라미터로 1000 epoch 까지 끌고 갈 예정이었음 — **v1 이 200 epoch 설계로 끝나서 long-tail 개선분을 못 본 게 있는지** 확인 목적. 40 epoch 까지의 추세는 v1 과 거의 일치 (자연스러운 결과).

## 재개 / 평가 명령

- **재개** (1000 epoch 까지): `python scripts/watchdog_sct.py` (last.pth 자동 resume)
- **현재 best.pth (epoch 33) 즉시 hold-out 60 평가**:
  ```bash
  python -m src.batch_infer --config configs/full_v2.yaml --ckpt outputs/full_v2/best.pth \
      --input-dir "outputs/inference input" --output-dir "outputs/inference output"
  python C:/Users/USER/Desktop/dev/nnUNet_translation/val.py \
      --pred-dir "outputs/inference output" --gt-image-dir outputs/new_gt \
      --save-dir outputs/metrics_v2_holdout60_epoch33
  ```

## 산출물 (이 세션)

- 학습 가중치: [`outputs/full_v2/best.pth`](./outputs/full_v2/best.pth) (epoch 33, val MAE 26.37) + `last.pth` (epoch 40)
- 학습 로그: [`outputs/full_v2/train.log`](./outputs/full_v2/train.log)
- 학습 history: [`outputs/full_v2/history.json`](./outputs/full_v2/history.json) (40 epoch 분)
- 학습 progress 그래프: [`outputs/full_v2/progress.png`](./outputs/full_v2/progress.png) (40 epoch 까지의 곡선)
- watchdog 로그: [`scripts/watchdog_v2_logs/watchdog.log`](./scripts/watchdog_v2_logs/watchdog.log)

---

# Hold-out 60 케이스 평가 — v2 best.pth (epoch 33, val MAE 26.37)

기록 시각: 2026-04-29 19:25
사용자 수동 trigger (post_train 의 자동 흐름과는 별개. 1000 epoch 완주 전 중간 평가).

## 입력 / 출력 / GT

| 항목 | 경로 |
|------|------|
| 입력 (CBCT) | `outputs/inference input/` (60 케이스) |
| 출력 (sCT) | `outputs/inference output/` (60 케이스, v1 의 .nii.gz 60개 덮어쓰기) |
| GT (CT) | `outputs/new_gt/` |
| 모델 | `outputs/full_v2/best.pth` (epoch 33, train-time val MAE 26.37) |
| 메트릭 JSON | `outputs/metrics_v2_holdout60_epoch33/metrics.json` |

## 결과 (n=60, val.py SynthRAD 표준)

| Metric | Mean | Std | Median | Min | Max |
|--------|-----:|----:|-------:|----:|----:|
| MAE_HU | 128.53 | 23.90 | 126.56 | 88.38 | 222.13 |
| PSNR_dB | 24.73 | 1.42 | 24.61 | — | — |
| SSIM | 0.9034 | 0.0363 | 0.9080 | 0.7752 | 0.9716 |
| MS_SSIM | 0.9029 | 0.0407 | 0.9096 | — | — |
| DICE_bone | 0.7017 | 0.0869 | 0.7076 | 0.5059 | 0.8586 |
| HD95_bone_mm | 5.10 | 2.47 | 4.85 | 2.00 | 16.91 |

`scale_ok` 통과: 60/60.

## v1 (full_0429 epoch 145) vs v2 (full_v2 epoch 33) 직접 비교

| Metric | v1 best (epoch 145, 148 ep run) | **v2 best (epoch 33, 40 ep 중단)** | 차이 |
|--------|--------------------------------:|----------------------------------:|-----:|
| MAE_HU | 127.99 | **128.53** | +0.54 (거의 동일) |
| PSNR_dB | 24.67 | **24.73** | +0.06 |
| SSIM | 0.9025 | **0.9034** | +0.0009 |
| MS_SSIM | 0.9021 | **0.9029** | +0.0008 |
| DICE_bone | 0.6999 | **0.7017** | +0.0018 |
| HD95_bone_mm | 5.01 | **5.10** | +0.09 |

> **관찰**: v2 가 epoch 33 (3.7배 적은 학습) 으로 v1 epoch 145 와 사실상 동일 성능. 200 epoch 이후의 long-tail 학습이 hold-out 일반화에 거의 기여 안 함을 시사. v1 이 200 epoch 까지 갔다고 해서 v2 보다 hold-out 에서 더 좋아질지는 불확실.

## Best 5 (lowest MAE_HU)

| Case | MAE_HU | SSIM | DICE_bone | HD95_bone_mm |
|------|-------:|-----:|----------:|-------------:|
| 2HNE072 | 88.38 | 0.9203 | 0.8586 | 2.00 |
| 2HNE010 | 91.21 | 0.9385 | 0.8210 | 3.16 |
| 2HNE038 | 93.48 | 0.9287 | 0.8394 | 2.83 |
| 2HNE036 | 94.51 | 0.8949 | 0.8448 | 2.00 |
| 2HNE064 | 98.15 | 0.8925 | 0.8145 | 3.00 |

## Worst 5 (highest MAE_HU)

| Case | MAE_HU | SSIM | DICE_bone | HD95_bone_mm |
|------|-------:|-----:|----------:|-------------:|
| 2HNE026 | 222.13 | 0.9290 | 0.6055 | 16.91 |
| 2HNE102 | 176.49 | 0.8572 | 0.5676 | 6.56 |
| 2HNE037 | 171.68 | 0.8913 | 0.5059 | 11.18 |
| 2HNE081 | 167.89 | 0.9368 | 0.5415 | 6.56 |
| 2HNE097 | 166.07 | 0.9079 | 0.5518 | 5.83 |

> Worst 5 케이스는 v1 평가 때와 거의 동일 (2HNE026, 2HNE102, 2HNE037 등) — 모델별로 일관되게 어려운 케이스. 데이터 자체 특성 (병변/금속 saturation 등) 의심.

## 산출물

- 인퍼런스 출력 60 × .nii.gz: `outputs/inference output/`
- 메트릭 JSON: `outputs/metrics_v2_holdout60_epoch33/metrics.json`
- 인퍼런스 stdout: `outputs/batch_infer_v2_epoch33.log`
- val.py stdout: `outputs/val_v2_epoch33.log`

## 50 epoch 단위 체크포인트 (사용자 요청 — 2026-04-29 18:00 추가)

| 항목 | 값 |
|------|----|
| 옵션 위치 | [configs/full_v2.yaml](./configs/full_v2.yaml) `training.save_every: 50` |
| train.py 동작 | val 단계에서 `(epoch+1) % save_every == 0` 일 때 `outputs/full_v2/checkpoint_epoch_NNNN.pth` (`NNNN` = epoch+1, 4자리 zero-pad) 저장 |
| best.pth / last.pth 와의 관계 | 별개. 매 epoch `last.pth` (덮어쓰기), best 갱신 시 `best.pth` (덮어쓰기), 50 마다 `checkpoint_epoch_NNNN.pth` (영구) — 모두 동시 저장 |
| 예상 총 파일 수 | 1000 epoch / 50 = **20 개** (`checkpoint_epoch_0050.pth` ... `checkpoint_epoch_1000.pth`) + best/last = 22 개 |
| 디스크 | 각 ~78 MB → 20 × 78 MB ≈ **1.5 GB** 추가 (충분히 여유) |
| 비활성화 | `save_every: 0` 으로 두면 snapshot 안 함 (default) |

## 변경된 학습 조건 (1000 epoch 으로 재시작)

| 항목 | 이전 (200 epoch run, epoch 5 에서 중단) | **현재 (1000 epoch run, fresh)** |
|------|----------------------------------------:|---------------------------------:|
| num_epochs (config) | 200 | **1000** |
| Watchdog NUM_EPOCHS | 200 | **1000** |
| Watchdog MAX_ATTEMPTS | 50 | 100 (긴 학습 대비) |
| CosineAnnealingLR T_max | 200 | **1000** (재생성) |
| 시작 lr | 1e-4 | 동일 |
| 다른 모든 설정 | 동일 (batch=8, L1, AMP, val_every=1) | 동일 |

## 인퍼런스 경로 (사용자 지정)

- 입력: `outputs/inference input/` (60 케이스 `*_0000.nii.gz`)
- 출력: `outputs/inference output/` ← v1 의 .nii.gz 60개 덮어쓰기 (v1 평가 메트릭은 [unet procceding.md](./unet%20procceding.md) 에 보존됨)
- GT: `outputs/new_gt/` (60 케이스 `<case>.nii.gz`)
- val 메트릭: `outputs/metrics_v2_holdout60/metrics.json`
- post_train 로그: `scripts/watchdog_v2_logs/post_train.log`
- inference / val stdout: `scripts/watchdog_v2_logs/predict_v2.log`, `val_v2.log`
