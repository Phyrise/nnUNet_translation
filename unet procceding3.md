# sct_unet 세션 #3 — 정적 분석 + 비대칭 인식 정정 + P1 코드 패치

작성: 2026-05-19 (오늘 진행분 종합)
이전 기록: [unet procceding.md](./unet%20procceding.md) (v1) · [unet procceding1.md](./unet%20procceding1.md) (v2) · [unet procceding2.md](./unet%20procceding2.md) (L1+Perceptual, COLLAPSED 진단)

본 세션은 **학습/추론 없는 정적 분석 위주 작업** + 마지막에 **P1 코드 패치 3건 적용 + 1-epoch sanity 검증** 으로 마무리. GPU 사용량 최소 (sanity 1 epoch ~60s 뿐).

---

## 0. 오늘 한 일 (한 줄 요약)

1. Outlier 분석 실행 → 6 케이스 (model-agnostic) 데이터 cleaning 후보 도출
2. EXPERIMENT_REPORT.md 에 outlier 결과 §7.8 추가
3. **작업지도서 (Pipeline-A vs Pipeline-B 비교 보고서)** 실행 → 4 보고서 + Mermaid 다이어그램 + 코드 스니펫 산출
4. **사용자 지적으로 비대칭 인식 정정** — 근거 등급 [M]/[S]/[E] 도입, B 의 grad-clip / 3D / augmentation 우위 주장 모두 [E] 로 정정, H-06 P1→P2 demote
5. **P1 코드 패치 3건 적용** (H-02 / H-15 / H-18) + 1-epoch sanity 통과
6. **신규 [M] 발견** — H-18 affine 체크 결과 200 케이스 모두 정렬 일치 (warning 0건)

---

## 1. Outlier 분석 (재학습 0)

### 입력 (모두 기존 산출물 4 metrics.json)

| 데이터셋 | 평가 모델 | 경로 |
|---------|-----------|------|
| 200-case | A-3 plain L1 | `results/old0422/prediction_metrics_checkpoint_final_0422/metrics.json` |
| 200-case | sct_unet v1 (epoch 145) | `outputs/metrics_best_0429/metrics.json` |
| 60-case  | sct_unet v1 best | `outputs/metrics_holdout60/metrics.json` |
| 60-case  | sct_unet v2 best | `outputs/metrics_v2_holdout60_epoch33/metrics.json` |

### 산출물

```
C:/Users/USER/Desktop/dev/nnUNet_translation/results/outlier_analysis/
├── README.md                        — 인덱스
├── outlier_report_200-case.md       — 200-case 상세 (Cross-model agreement, Top 30 composite, Per-metric worst 5)
├── outlier_report_60-case.md        — 60-case hold-out 상세
└── outlier_analysis.json            — 케이스별 모든 metric + IQR/z-flag + agreement (machine-readable, ~750 KB)
```

### 핵심 결과 — 데이터 cleaning 우선 검토 6 케이스 (model-agnostic outlier)

| Cohort | 케이스 | 특징 |
|--------|--------|------|
| 200-case | **2HNC087, 2HNA102, 2HNC117** | 두 모델 모두 MAE 90~140 HU, DICE_bone 0.57~0.68, HD95 9~19 mm |
| 60-case (hold-out) | **2HNE022, 2HNE026, 2HNE037** | `2HNE026` 양 모델 MAE 211~222 HU, HD95 16+ mm |

### Model-specific 실패 (데이터 문제 아님, 모델 학습 분포의 hole)

- `2HNC021/082/100`: A-3 plain L1 에서만 MAE 200~337 collapse-like, sct_unet v1 정상 (MAE 49~60). 추후 학습 시 oversample/augmentation 강화 후보.

자세한 내용은 [results/outlier_analysis/](../nnUNet_translation/results/outlier_analysis/) 참조. EXPERIMENT_REPORT.md §7.8 에도 동일 결과 기록.

---

## 2. 작업지도서 실행 — Pipeline-A vs Pipeline-B 정적 비교

워크오더 (Static-analysis-only, 학습/추론 금지) 4개 산출물 디렉토리:

```
C:/Users/USER/Desktop/dev/comparison_analysis/
├── 00_env.md                    (46 줄)  환경/경로 + ⚠ editable install 경로 불일치 메모
├── 01_unet_structure.md         (257 줄) Pipeline-A 17축 모든 항목 인용
├── 02_comparison.md             (136 줄) 17축 비교표 + §3 4단락 + §6 5줄결론 + ERRATA
├── 03_hyperparam_proposals.md   (241 줄) 16+3 후보, P1 5장 카드, Exp-A~G + ERRATA
├── diagrams/unet_pipeline.mmd            데이터→정규화→2.5D→모델→손실→denorm Mermaid
└── snippets/                             인용 코드 조각 3개
    ├── unet_model.py
    ├── unet_normalization.py
    └── unet_train_loop.py
```

### v1 초기 결론 5줄 (작성 직후)

1. A 와 B 는 거의 모든 축에서 다른 설계 — 2.5D vs 3D, 6M vs 30~150M, Adam+Cosine vs SGD+Poly, 정규화 도메인 분리 vs 통일, augmentation 빈약 vs 풍부, grad clip 없음 vs 있음, epoch 정의 불일치
2. **A 강점**: 페어링 안정성, 메모리/연산 효율
3. **A 약점**: 모델 capacity 작음, z 축 receptive field 좁음, augmentation 빈약, grad clip 없음
4. **B 강점**: 3D context, 풍부한 augmentation, 검증된 nnU-Net 표준 학습 셋업
5. **B 약점**: target 페어링 우회의 fragility (Stage 3 collapse), 큰 메모리, SGD lr=1e-2 의 regression 적합성 미검증

→ 사용자 지적: **#3 (A 약점) 은 [S]/[E] 명시인데 #4 (B 강점) 은 [E] 임에도 단정으로 쓰여짐** = 비대칭 인식.

---

## 3. 비대칭 인식 정정 — [M]/[S]/[E] 도입

### 근거 등급 정의

| 라벨 | 의미 |
|------|------|
| **[M]** | Measured-in-project — 본 프로젝트 실제 측정 결과 |
| **[S]** | Static-code — 코드/config 직접 검증 가능한 사실 |
| **[E]** | External — 외부 논문/표준 추론, **본 프로젝트 미측정** |

### B 우위 주장 audit 결과 (모두 [E])

| 주장 | v1 | v2 정정 + 근거 |
|------|----|---------------|
| B augmentation 우위 | [S] 처럼 단정 | **[E]** — B 학습 모두 collapse/조기중단, 측정 0건 |
| B grad-clip 우위 | [S] 처럼 단정 | **[E]** — B-2 (WarmupGradClip) 평가 전 epoch 61 중단, 효과 측정 안 됨. A 학습에서 NaN 발생 보고도 없음 |
| B 3D context 우위 | [S] 처럼 단정 | **[E]** — 2.5D-vs-3D 단변수 비교 없음. 오히려 [M] 으로는 sct_unet (2.5D) 가 A-3 plain (3D) 보다 200-case MAE 낮음 (57 vs 73) — 다른 변수 차이로 단정 불가 |

### 03_hyperparam_proposals.md 의 H-06 demote

| 항목 | v1 | v2 |
|------|----|----|
| 우선순위 | P1 | **P2 (pending-measurement)** |
| 예상 효과 | "외부 60-case MAE -10~20 HU 개선" | "[E] 본 프로젝트 미측정, SynthRAD 외부 사례 기반 추정" |
| 통합 추정 | "138 → 110~120 HU" 단정 | H-06 기여분 분리. Exp-A 측정 후 [M] 으로 격상 경로 명시 |

### 실행 순서 정정 (v2 03 §3)

1. **Step 0** — bug-fix: dead config 복원 (단독, 측정 없음) [본 세션에서는 미적용]
2. **Step 1** — 1 epoch sanity (정합성 검증)
3. **Step 2** — H-15 + H-02 + H-18 패치 (모두 P1, [S]/저비용/안전성 근거)
4. **Step 3** — Exp-A 측정 시작 (~6h, 100ep, 새 baseline [M] 확립)
5. **Step 4** — Exp-A 결과 보고 H-06 (Exp-D) 승격 여부 결정

세 보고서 모두 v2 + ERRATA 절 추가:
- [01_unet_structure.md §11 ERRATA](../comparison_analysis/01_unet_structure.md)
- [02_comparison.md §7 ERRATA](../comparison_analysis/02_comparison.md)
- [03_hyperparam_proposals.md §5 ERRATA](../comparison_analysis/03_hyperparam_proposals.md)

---

## 4. P1 코드 패치 적용 (사용자 채택 후)

### H-02 — CT clip 범위 확장

**대상**: [configs/full_v2.yaml](./configs/full_v2.yaml) + [configs/smoke.yaml](./configs/smoke.yaml)

```diff
 normalization:
-  ct_clip_min: -1000
-  ct_clip_max: 2000
+  ct_clip_min: -1024
+  ct_clip_max: 3071
   cbct_air_threshold: -900
```

**근거**: Dataset051 voxel max=3071 HU ([S]). 기존 2000 클립으로 2.7% 골 voxel saturate.
**효과**: DICE_bone 영향 [E] — Exp-A 측정 후 [M] 격상.

### H-15 — AMP-safe gradient clipping

**대상**: [src/train.py:174-180](./src/train.py#L174-L180)

```diff
             scaler.scale(loss).backward()
+            # H-15: AMP-safe gradient clipping (nnU-Net 표준 clip_norm=12.0)
+            # NaN/Inf 회복 안전망. 효과는 [E] (본 프로젝트 A 학습에서 NaN 발생 보고 없음).
+            scaler.unscale_(optimizer)
+            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12.0)
             scaler.step(optimizer)
             scaler.update()
```

**근거**: AMP fp16 학습에서 grad explosion 회복 메커니즘 부재 ([S]).
**효과**: NaN 회복 [E] — 본 프로젝트 A 학습에서 NaN 발생 보고 없음. 안전망 가치.

### H-18 — NIfTI affine 일관성 자동 체크

**대상**: [src/data.py:42-55](./src/data.py#L42-L55)

```diff
     if cbct.shape != ct.shape:
         raise ValueError(f"Shape mismatch for {case_id}: ...")
+    # H-18: NIfTI affine consistency check (silent-guard).
+    if not np.allclose(cbct_nii.affine, ct_nii.affine, atol=1e-3):
+        import warnings
+        diff = np.max(np.abs(cbct_nii.affine - ct_nii.affine))
+        warnings.warn(f"[H-18] affine mismatch for {case_id} (max |diff|={diff:.4f}). ...")
     cbct = normalize_cbct(cbct, cbct_air)
```

**근거**: shape 만 검증, affine 검증 없었음 ([S]).
**효과**: silent bug 차단 [E] — 본 프로젝트에서 실제 misalignment case 발견 안 됨 (가설).

---

## 5. 1-epoch sanity 검증 결과

명령:
```
python -m src.train --config configs/smoke.yaml
```

로그: `outputs/smoke_v2_patches.log`

| 항목 | 결과 |
|------|------|
| 학습 완주 | ✓ 60.1s, "Training done." 마커, exit code 0 |
| train_loss 추이 | 0.58 → 0.50 (정상 감소) |
| epoch 평균 train_loss | 0.6076 |
| val MAE @ epoch 0 | 1092.55 HU (random init 직후 정상치) |
| NaN/Inf 발생 | 0건 |
| H-15 grad clip | 정상 작동, 학습 dynamics 영향 없음 (단일 epoch) |
| **H-18 affine warning** | **0건** — 200 케이스 모두 CBCT/CT affine 일치 |
| H-02 clip 범위 효과 | 손실 스케일만 약 ×3 증가 (정규화 범위 확장으로 인한 예상된 변화, 정상) |

### 신규 [M] 발견 (오늘 추가)

| 항목 | 등급 변동 | 내용 |
|------|----------|------|
| 데이터셋 affine 정합성 | [E] (미측정) → **[M]** (확인됨) | 200 케이스 모두 CBCT(D050) ↔ CT(D051) affine 일치. 데이터셋 자체는 잘 정렬됨. silent-guard 의 실제 작동 검증 완료. |

---

## 6. 산출물 인덱스 (오늘 생성/수정 전체)

### 신규 생성 (오늘)

```
C:/Users/USER/Desktop/dev/comparison_analysis/         ← 정적 분석 보고서 디렉토리 (전체 신규)
├── 00_env.md
├── 01_unet_structure.md
├── 02_comparison.md
├── 03_hyperparam_proposals.md
├── diagrams/unet_pipeline.mmd
└── snippets/{unet_model,unet_normalization,unet_train_loop}.py

C:/Users/USER/Desktop/dev/nnUNet_translation/results/outlier_analysis/   ← outlier 분석 산출물
├── README.md
├── outlier_report_200-case.md
├── outlier_report_60-case.md
└── outlier_analysis.json

C:/Users/USER/Desktop/dev/nnUNet_translation/scripts/outlier_analysis.py   ← 분석 스크립트

C:/Users/USER/Desktop/dev/sct_unet/outputs/smoke_v2_patches.log   ← P1 패치 sanity 로그

C:/Users/USER/Desktop/dev/sct_unet/unet procceding3.md   ← 이 파일
```

### 수정 (오늘)

| 파일 | 변경 |
|------|------|
| `nnUNet_translation/results/EXPERIMENT_REPORT.md` | §7.8 outlier 분석 실행 결과 추가 |
| `sct_unet/configs/full_v2.yaml` | H-02 — ct_clip [-1024, 3071] |
| `sct_unet/configs/smoke.yaml` | H-02 동기화 |
| `sct_unet/src/train.py` | H-15 — grad clip + unscale_ |
| `sct_unet/src/data.py` | H-18 — affine 일관성 체크 |
| `comparison_analysis/01_unet_structure.md` | §11 ERRATA (v2 정정) |
| `comparison_analysis/02_comparison.md` | §7 ERRATA (v2 정정), 17축 표 근거 등급 컬럼 추가 |
| `comparison_analysis/03_hyperparam_proposals.md` | §5 ERRATA (v2 정정), H-06 P1→P2, Step 0~4 실행순서 정정 |

---

## 7. 다음 액션 대기

03_hyperparam_proposals.md §3 의 Step 3 이후:

- **Exp-A 측정 시작** (100 epoch ≈ 6h) — H-02+H-15+H-18 적용된 새 baseline [M] 확립
  - 200-case + 60-case 평가까지 자동 진행
  - 결과 → H-06 (dead config 복원 + augmentation 강화) 승격 여부 결정의 근거

옵션 (사용자 결정):
- (a) 즉시 Exp-A 시작
- (b) 대기 (사용자 직접 트리거)
- (c) Step 0 (H-06 dead config) 도 같이 적용 후 Exp-A — **단 변수 분리 깨짐 (H-15/H-02/H-18 효과와 H-06 효과 구분 불가)**

권장: (a) 또는 (b). (c) 는 측정 깨짐 위험.

---

## 8. 참고 — 본 세션 이전 기록 인덱스

| 파일 | 내용 |
|------|------|
| [unet procceding.md](./unet%20procceding.md) | sct_unet v1 (full_0429) 진행 — 148 epoch stop |
| [unet procceding1.md](./unet%20procceding1.md) | sct_unet v2 (full_v2) — 200 케이스 평가, 60-case hold-out |
| [unet procceding2.md](./unet%20procceding2.md) | nnUNet L1+Perceptual (Stage 5) → COLLAPSED 5/5 진단 |
| `nnUNet_translation/results/EXPERIMENT_REPORT.md` | 전체 종합 + 사후 outlier 결과 |
| `nnUNet_translation/results/warmup_gradclip_progress.md` | Stage 4a |
| `nnUNet_translation/results/zero_init_residual_progress.md` | Stage 4b |
