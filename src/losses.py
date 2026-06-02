"""Regression losses for sCT synthesis."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class L1Loss(nn.Module):
    def forward(self, pred, target):
        return F.l1_loss(pred, target)


class L1SSIMLoss(nn.Module):
    """L1 + ssim_weight * (1 - SSIM). 2D only.

    SSIM은 monai.metrics 기반이 가장 안전하지만, 학습용으로는 가벼운 직접 구현으로
    의존성을 최소화한다.
    """

    def __init__(self, ssim_weight: float = 0.1, win_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.win_size = win_size
        self.sigma = sigma
        self.register_buffer("_kernel", self._gaussian_kernel(win_size, sigma))

    @staticmethod
    def _gaussian_kernel(win: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(win).float() - (win - 1) / 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        k2d = g[:, None] @ g[None, :]
        return k2d.view(1, 1, win, win)

    def _ssim(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x, y: (B, 1, H, W) in [-1, 1]; treat data range as 2.0
        kernel = self._kernel.to(x.dtype).to(x.device)
        pad = self.win_size // 2
        mu_x = F.conv2d(x, kernel, padding=pad)
        mu_y = F.conv2d(y, kernel, padding=pad)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sig_x2 = F.conv2d(x * x, kernel, padding=pad) - mu_x2
        sig_y2 = F.conv2d(y * y, kernel, padding=pad) - mu_y2
        sig_xy = F.conv2d(x * y, kernel, padding=pad) - mu_xy
        L = 2.0
        c1 = (0.01 * L) ** 2
        c2 = (0.03 * L) ** 2
        ssim_map = ((2 * mu_xy + c1) * (2 * sig_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sig_x2 + sig_y2 + c2))
        return ssim_map.mean()

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        ssim = self._ssim(pred, target)
        return l1 + self.ssim_weight * (1.0 - ssim)


def build_loss(cfg: dict) -> nn.Module:
    name = cfg["training"].get("loss", "l1")
    if name == "l1":
        return L1Loss()
    if name == "l1_ssim":
        return L1SSIMLoss(ssim_weight=cfg["training"].get("ssim_weight", 0.1))
    raise ValueError(f"Unknown loss: {name}")
