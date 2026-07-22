from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


APP_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = APP_ROOT / "assets"


def test_contact_fold_icon_assets_are_complete() -> None:
    expected = (
        ASSET_DIR / "touch_system_icon_source.png",
        ASSET_DIR / "touch_system_icon.png",
        ASSET_DIR / "touch_system_icon.ico",
        ASSET_DIR / "touch_system_icon.svg",
        ASSET_DIR / "touch_system_icon_preview.png",
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in expected)


def test_master_icon_has_transparency_safe_margin_and_three_color_signature() -> None:
    image = Image.open(ASSET_DIR / "touch_system_icon.png").convert("RGBA")
    assert image.size == (1024, 1024)
    assert image.getpixel((0, 0))[3] == 0

    bbox = image.getbbox()
    assert bbox is not None
    assert min(bbox[0], bbox[1], 1024 - bbox[2], 1024 - bbox[3]) >= 35

    pixels = np.asarray(image)
    alpha = pixels[:, :, 3] > 32
    cyan = alpha & (pixels[:, :, 2] > 150) & (pixels[:, :, 1] > 120) & (pixels[:, :, 0] < 160)
    teal = alpha & (pixels[:, :, 0] < 70) & (pixels[:, :, 1] < 165) & (pixels[:, :, 2] < 175)
    coral = alpha & (pixels[:, :, 0] > 180) & (pixels[:, :, 1] < 190) & (pixels[:, :, 2] < 180)
    assert cyan.sum() > 50_000
    assert teal.sum() > 5_000
    assert coral.sum() > 10_000


def test_ico_contains_optically_corrected_micro_frames() -> None:
    icon = Image.open(ASSET_DIR / "touch_system_icon.ico")
    expected_sizes = {16, 20, 24, 32, 40, 48, 64, 96, 128, 256}
    actual_sizes = {width for width, height in icon.ico.sizes() if width == height}
    assert expected_sizes <= actual_sizes

    micro = np.asarray(icon.ico.getimage((16, 16)).convert("RGBA"))
    alpha = micro[:, :, 3] > 32
    cyan = alpha & (micro[:, :, 2] > 135) & (micro[:, :, 1] > 115) & (micro[:, :, 0] < 125)
    teal = alpha & (micro[:, :, 0] < 45) & (micro[:, :, 1] < 150) & (micro[:, :, 2] < 155)
    coral = alpha & (micro[:, :, 0] > 180) & (micro[:, :, 1] < 180) & (micro[:, :, 2] < 170)
    assert alpha.sum() >= 45
    assert cyan.sum() >= 20
    assert teal.sum() >= 3
    assert coral.sum() >= 3


def test_desktop_build_uses_the_contact_fold_ico() -> None:
    spec = (APP_ROOT / "desktop_launcher.spec").read_text(encoding="utf-8")
    assert 'assets" / "touch_system_icon.ico"' in spec
