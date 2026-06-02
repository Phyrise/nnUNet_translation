"""2.5D paired CBCT→CT dataset.

볼륨 단위로 NIfTI를 메모리에 올려두고, 슬라이스 인덱스를 샘플로 노출한다.
- 입력: CBCT slice z 와 인접 슬라이스 (총 num_adjacent 채널, 보통 3)
- 타겟: CT slice z (1채널)
"""
from __future__ import annotations
import os
import glob
from dataclasses import dataclass

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

from .utils import normalize_cbct, normalize_ct


@dataclass
class CaseVolume:
    case_id: str
    cbct: np.ndarray  # (H, W, D), normalized
    ct: np.ndarray    # (H, W, D), normalized to [-1, 1]
    affine: np.ndarray
    header: object


def _list_cases(cbct_dir: str, ct_dir: str) -> list[str]:
    # C1: 페어링 검증 강화 — CBCT/CT 가 동일 case_id 의 동일 dir 구조 보장
    cbct_files = sorted(glob.glob(os.path.join(cbct_dir, "*_0000.nii.gz")))
    cases = []
    for p in cbct_files:
        cid = os.path.basename(p).replace("_0000.nii.gz", "")
        ct_path = os.path.join(ct_dir, f"{cid}_0000.nii.gz")
        if os.path.isfile(ct_path):
            cases.append(cid)
    # C1: 중복 case_id 차단 (paranoid — _list_cases 자체에서는 발생 안 함, sanity)
    if len(cases) != len(set(cases)):
        dupes = [c for c in cases if cases.count(c) > 1]
        raise RuntimeError(f"[C1] Duplicate case_id detected in _list_cases: {set(dupes)}")
    return sorted(cases)


def load_case(cbct_dir: str, ct_dir: str, case_id: str,
              ct_clip: tuple[float, float], cbct_air: float) -> CaseVolume:
    # C1: case_id 가 file path 와 일치하는지 paranoid assert
    cbct_path = os.path.join(cbct_dir, f"{case_id}_0000.nii.gz")
    ct_path = os.path.join(ct_dir, f"{case_id}_0000.nii.gz")
    cbct_basename_cid = os.path.basename(cbct_path).replace("_0000.nii.gz", "")
    ct_basename_cid = os.path.basename(ct_path).replace("_0000.nii.gz", "")
    if cbct_basename_cid != case_id or ct_basename_cid != case_id:
        raise RuntimeError(
            f"[C1] case_id mismatch in load_case: arg={case_id}, "
            f"cbct path={cbct_basename_cid}, ct path={ct_basename_cid}"
        )
    cbct_nii = nib.load(cbct_path)
    ct_nii = nib.load(ct_path)
    cbct = cbct_nii.get_fdata().astype(np.float32)
    ct = ct_nii.get_fdata().astype(np.float32)
    if cbct.shape != ct.shape:
        raise ValueError(f"Shape mismatch for {case_id}: CBCT {cbct.shape} vs CT {ct.shape}")
    # H-18: NIfTI affine consistency check (silent-guard).
    # 효과 [E] — 본 프로젝트에서 실제 misalignment case 발견 안 됨.
    # atol=1e-3: spacing 단위 mm 의 1/1000 (sub-micron) — 안전한 tolerance.
    if not np.allclose(cbct_nii.affine, ct_nii.affine, atol=1e-3):
        import warnings
        diff = np.max(np.abs(cbct_nii.affine - ct_nii.affine))
        warnings.warn(
            f"[H-18] affine mismatch for {case_id} (max |diff|={diff:.4f}). "
            f"CBCT/CT 정렬이 불일치 → voxel-wise loss 의미가 약함. "
            f"skip 또는 데이터 재정합 검토 필요."
        )
    cbct = normalize_cbct(cbct, cbct_air)
    ct = normalize_ct(ct, ct_clip[0], ct_clip[1])
    return CaseVolume(case_id, cbct, ct, cbct_nii.affine, cbct_nii.header)


class PairedSliceDataset(Dataset):
    """2.5D paired slice dataset for training.

    Each sample: (CBCT 3-channel slice stack, CT 1-channel slice).
    """

    def __init__(self, cases: list[CaseVolume], num_adjacent: int = 3,
                 patch_size: tuple[int, int] | None = (256, 256),
                 augment: bool = False, aug_cfg: dict | None = None):
        assert num_adjacent % 2 == 1, "num_adjacent must be odd"
        # C1: case_id unique 검증 — 중복은 페어링 혼선 위험
        case_ids = [c.case_id for c in cases]
        if len(case_ids) != len(set(case_ids)):
            dupes = [cid for cid in case_ids if case_ids.count(cid) > 1]
            raise RuntimeError(f"[C1] Duplicate case_id in PairedSliceDataset: {set(dupes)}")
        # C1: 각 case 의 cbct/ct shape 일치 paranoid assert (load_case 가 이미 확인하나 sanity)
        for c in cases:
            if c.cbct.shape != c.ct.shape:
                raise RuntimeError(
                    f"[C1] CaseVolume shape mismatch for {c.case_id}: "
                    f"cbct {c.cbct.shape} vs ct {c.ct.shape}"
                )
        self.cases = cases
        self.num_adjacent = num_adjacent
        self.half = num_adjacent // 2
        self.patch_size = patch_size
        self.augment = augment
        self.aug_cfg = aug_cfg or {}
        # build (case_idx, z) index
        self.index: list[tuple[int, int]] = []
        for ci, c in enumerate(cases):
            D = c.cbct.shape[-1]
            for z in range(D):
                self.index.append((ci, z))

    def __len__(self):
        return len(self.index)

    def _get_stack(self, vol: np.ndarray, z: int) -> np.ndarray:
        D = vol.shape[-1]
        idxs = [min(max(z + dz, 0), D - 1) for dz in range(-self.half, self.half + 1)]
        # vol: (H, W, D) → stack on axis 0 to get (C, H, W)
        return np.stack([vol[..., i] for i in idxs], axis=0)

    def _random_crop(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.patch_size is None:
            return x, y
        ph, pw = self.patch_size
        _, H, W = x.shape
        if H < ph or W < pw:
            pad_h = max(0, ph - H)
            pad_w = max(0, pw - W)
            x = np.pad(x, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
            y = np.pad(y, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")
            _, H, W = x.shape
        top = np.random.randint(0, H - ph + 1)
        left = np.random.randint(0, W - pw + 1)
        return x[:, top:top + ph, left:left + pw], y[:, top:top + ph, left:left + pw]

    def _augment(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.aug_cfg
        # horizontal flip
        if np.random.rand() < cfg.get("rand_flip_prob", 0.0):
            x = x[:, :, ::-1].copy()
            y = y[:, :, ::-1].copy()
        # gaussian noise on CBCT only
        if np.random.rand() < cfg.get("rand_noise_prob", 0.0):
            std = cfg.get("rand_noise_std", 0.05)
            x = x + np.random.randn(*x.shape).astype(np.float32) * std
        return x, y

    def __getitem__(self, idx: int) -> dict:
        ci, z = self.index[idx]
        case = self.cases[ci]
        # C1: paranoid pairing assert — case.cbct 와 case.ct 가 같은 CaseVolume 객체에서 추출됨을 보장
        # (load_case 가 CaseVolume(case_id, cbct, ct, ...) 로 single-object 페어링하므로 본 assert 는 사실상 항상 True.
        #  Stage 5 collapse 같은 dataloader 페어링 누락 (nnUNet MRCT 의 D050/D051 sync) 와 다른 layer)
        if case.cbct.shape[-1] != case.ct.shape[-1]:
            raise RuntimeError(
                f"[C1] runtime pairing mismatch at idx={idx} case={case.case_id}: "
                f"cbct D={case.cbct.shape[-1]} vs ct D={case.ct.shape[-1]}"
            )
        x = self._get_stack(case.cbct, z)            # (C, H, W)
        y = case.ct[..., z][None, ...]                # (1, H, W)
        x, y = self._random_crop(x, y)
        if self.augment:
            x, y = self._augment(x, y)
        return {
            "input": torch.from_numpy(np.ascontiguousarray(x)).float(),
            "target": torch.from_numpy(np.ascontiguousarray(y)).float(),
            "case_id": case.case_id,
            "z": z,
        }


def build_datasets(cfg: dict) -> tuple[PairedSliceDataset, PairedSliceDataset, list[CaseVolume], list[CaseVolume]]:
    d = cfg["data"]
    n = cfg["normalization"]
    s = cfg["slicing"]
    cases = _list_cases(d["cbct_dir"], d["ct_dir"])
    if not cases:
        raise RuntimeError(f"No paired cases found in {d['cbct_dir']} / {d['ct_dir']}")
    val_ids = set(d.get("val_cases") or [])
    if not val_ids:
        # fallback: last 3 cases as val
        val_ids = set(cases[-3:])
    train_ids = [c for c in cases if c not in val_ids]
    val_ids_list = [c for c in cases if c in val_ids]

    # C1: train / val intersection = 빈 set 인지 assert (leakage 차단)
    intersection = set(train_ids) & set(val_ids_list)
    if intersection:
        raise RuntimeError(
            f"[C1] train ∩ val intersection 발견 (leakage): {sorted(intersection)}"
        )

    ct_clip = (n["ct_clip_min"], n["ct_clip_max"])
    cbct_air = n["cbct_air_threshold"]

    train_vols = [load_case(d["cbct_dir"], d["ct_dir"], cid, ct_clip, cbct_air) for cid in train_ids]
    val_vols = [load_case(d["cbct_dir"], d["ct_dir"], cid, ct_clip, cbct_air) for cid in val_ids_list]

    train_ds = PairedSliceDataset(
        train_vols,
        num_adjacent=s["num_adjacent"],
        patch_size=tuple(s["patch_size"]),
        augment=True,
        aug_cfg=cfg["augmentation"],
    )
    val_ds = PairedSliceDataset(
        val_vols,
        num_adjacent=s["num_adjacent"],
        patch_size=None,  # full slice for val
        augment=False,
    )
    return train_ds, val_ds, train_vols, val_vols
