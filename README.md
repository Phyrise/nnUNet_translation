# sCT U-Net (MONAI 2.5D)

CBCT → sCT (synthetic CT) 합성용 2.5D U-Net. 분할용 U-Net을 회귀(image-to-image)로 개조한 독립 fork. nnUNet 의존성 없음.

자세한 설계와 진행 상황은 [PLAN.md](PLAN.md) 참고.

## 1. 환경 준비 (GPU 머신)

```bash
# (권장) 새 conda env
conda create -n sct python=3.11 -y
conda activate sct

# PyTorch — CUDA 버전에 맞춰 설치
# 예: CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 나머지 의존성
pip install -r requirements.txt
```

CUDA 인식 확인:
```python
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. 데이터 경로 확인

[`configs/default.yaml`](configs/default.yaml)의 `data.cbct_dir`, `data.ct_dir`만 GPU 머신 경로에 맞게 수정. 데이터 포맷:

```
<cbct_dir>/Case_001_0000.nii.gz   ← CBCT
<ct_dir>/Case_001_0000.nii.gz     ← CT (같은 좌표계, 정합 완료)
```

기본 split: `Case_012/013/014` 가 val, 나머지가 train.

## 3. 학습

```bash
# sct_unet/ 루트에서
python -m src.train --config configs/default.yaml
# 또는
./scripts/train.sh
```

산출물 (`outputs/`):
- `best.pth`, `last.pth` — 체크포인트
- `train.log` — 텍스트 로그
- `tb/` — TensorBoard (`tensorboard --logdir outputs/tb`)

## 4. 추론

```bash
python -m src.infer --config configs/default.yaml \
    --ckpt outputs/best.pth \
    --input /path/to/CBCT.nii.gz \
    --output /path/to/sCT.nii.gz
```

출력: int16 HU 단위 NIfTI, 입력의 affine/header 보존.

## 5. 스모크 테스트 (1 epoch만)

`configs/default.yaml`을 임시로:
```yaml
training:
  num_epochs: 1
  batch_size: 2
```
로 바꾸고 학습 → 추론까지 끝까지 도는지 확인 후 원복.

## 6. 디렉토리 구조

```
sct_unet/
├── PLAN.md                  설계/진행상황
├── README.md                이 문서
├── requirements.txt
├── configs/default.yaml
├── src/
│   ├── data.py              2.5D paired NIfTI dataset
│   ├── model.py             MONAI UNet 래퍼
│   ├── losses.py            L1 / L1+SSIM
│   ├── utils.py             정규화/시드/로거
│   ├── train.py
│   └── infer.py
├── scripts/                 train.sh/.bat, infer.sh/.bat
└── outputs/                 (런타임 생성)
```

## 7. 분할 U-Net과의 차이 (개조 포인트)

1. `out_channels=1`, softmax/argmax 제거 — 연속값(HU) 회귀
2. Loss: CE/Dice → **L1** (옵션 L1+SSIM)
3. Dataset: (image, mask) → **(CBCT, CT)** paired NIfTI
4. CT [-1000, 2000] HU → [-1, 1] 학습용, 출력 시 HU로 복원
5. 추론: 슬라이스별 sliding window → 볼륨 재조립, affine 보존
