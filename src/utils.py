import os
import random
import logging
import numpy as np
import torch
import yaml


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def normalize_ct(vol: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    """HU → [-1, 1] linear mapping after clipping."""
    v = np.clip(vol, clip_min, clip_max).astype(np.float32)
    return 2.0 * (v - clip_min) / (clip_max - clip_min) - 1.0


def denormalize_ct(vol: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    """[-1, 1] → HU."""
    v = np.clip(vol, -1.0, 1.0).astype(np.float32)
    return (v + 1.0) * 0.5 * (clip_max - clip_min) + clip_min


def normalize_cbct(vol: np.ndarray, air_threshold: float) -> np.ndarray:
    """Per-volume z-score over foreground (intensity > air_threshold)."""
    v = vol.astype(np.float32)
    mask = v > air_threshold
    if mask.sum() < 100:
        mask = np.ones_like(v, dtype=bool)
    mu = v[mask].mean()
    sd = v[mask].std()
    if sd < 1e-6:
        sd = 1.0
    return (v - mu) / sd
