"""MONAI UNet wrapper for CBCT→CT synthesis (linear output, regression)."""
from monai.networks.nets import UNet


def build_unet(cfg: dict) -> UNet:
    m = cfg["model"]
    return UNet(
        spatial_dims=m["spatial_dims"],
        in_channels=m["in_channels"],
        out_channels=m["out_channels"],
        channels=tuple(m["channels"]),
        strides=tuple(m["strides"]),
        num_res_units=m.get("num_res_units", 2),
        dropout=m.get("dropout", 0.0),
    )
