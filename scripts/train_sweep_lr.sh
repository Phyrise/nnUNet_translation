#!/usr/bin/env bash
# train_sweep_lr.sh — sct_unet full_v3_paired_fix 의 lr sweep 자동 실행
# C3: Learning Rate sweep {1e-3, 5e-4, 1e-4, 5e-5}
# 작성: 2026-05-27 / amed plan unet0601.md § 3.3 참조
#
# 사용법:
#   bash scripts/train_sweep_lr.sh              # 4 lr 순차 실행
#   bash scripts/train_sweep_lr.sh --dry-run    # 명령만 echo, 실행 안 함
#
# 전제: P_dino_pca / 트랙 2 학습 종료 후 (5/28 09:00 이후 권장)
# 비용: 100 epoch × 4 lr ≈ 24h + post_train 자동 평가 ~2h = ~26h

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJ_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJ_ROOT"

CONFIG="configs/full_v3_paired_fix.yaml"
LRS=("1e-3" "5e-4" "1e-4" "5e-5")
DRY_RUN=0

# CLI 옵션
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --help|-h)
            echo "Usage: $0 [--dry-run]"
            echo "  --dry-run : echo commands only, do not execute"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg" >&2
            exit 2
            ;;
    esac
done

# Python 환경 확인
PYTHON="${PYTHON:-python}"
if [ ! -x "$(command -v "$PYTHON")" ]; then
    # Anaconda nnunet 환경 fallback
    PYTHON="C:/Users/USER/anaconda3/envs/nnunet/python.exe"
    if [ ! -x "$PYTHON" ]; then
        echo "Python not found. Set PYTHON env var or activate nnunet env." >&2
        exit 1
    fi
fi

echo "=== sct_unet lr sweep ==="
echo "  config: $CONFIG"
echo "  lrs: ${LRS[*]}"
echo "  python: $PYTHON"
echo "  dry-run: $DRY_RUN"
echo ""

# config 존재 확인
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config not found: $CONFIG" >&2
    exit 1
fi

# GPU 점유 경고 (P_dino_pca 등 진행 중이면)
if command -v nvidia-smi &>/dev/null; then
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$GPU_MEM" ] && [ "$GPU_MEM" -gt 1024 ]; then
        echo "WARNING: GPU 사용 중 (memory.used=${GPU_MEM} MiB). 트랙 2 학습 종료 후 권장."
        if [ "$DRY_RUN" -eq 0 ]; then
            read -p "  Continue anyway? [y/N]: " yn
            case "$yn" in
                [yY]*) ;;
                *) echo "Aborted."; exit 0 ;;
            esac
        fi
    fi
fi

# sweep 실행
for LR in "${LRS[@]}"; do
    OUT_DIR="outputs/full_v3_lr_${LR}"
    echo "[$(date +%H:%M:%S)] === lr=${LR} → ${OUT_DIR} ==="
    CMD=(
        "$PYTHON" -m src.train
        --config "$CONFIG"
        --override "training.lr=${LR}"
        --override "training.out_dir=${OUT_DIR}"
    )
    echo "  cmd: ${CMD[*]}"
    if [ "$DRY_RUN" -eq 0 ]; then
        # 결과 디렉토리 미리 생성
        mkdir -p "$OUT_DIR"
        # 학습 실행 — log 도 별도 저장
        if "${CMD[@]}" 2>&1 | tee "${OUT_DIR}/train_sweep.log"; then
            echo "[$(date +%H:%M:%S)] lr=${LR} 완료"
        else
            EXIT_CODE=$?
            echo "[$(date +%H:%M:%S)] lr=${LR} 실패 (exit=$EXIT_CODE). 다음 lr 로 진행."
        fi
    fi
    echo ""
done

echo "=== sweep 완료 ==="
echo "결과 디렉토리:"
for LR in "${LRS[@]}"; do
    OUT_DIR="outputs/full_v3_lr_${LR}"
    if [ -f "${OUT_DIR}/best.pth" ]; then
        BEST_MAE=$(grep -oE 'new best MAE: [0-9.]+' "${OUT_DIR}/train.log" 2>/dev/null | tail -1)
        echo "  lr=${LR}: ${OUT_DIR}/best.pth (${BEST_MAE:-no best yet})"
    else
        echo "  lr=${LR}: ${OUT_DIR} (best.pth 부재)"
    fi
done
