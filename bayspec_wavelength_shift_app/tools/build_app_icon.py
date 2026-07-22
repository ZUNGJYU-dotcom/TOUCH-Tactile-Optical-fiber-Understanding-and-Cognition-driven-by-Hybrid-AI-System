from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import (
    Image,
    ImageChops,
    ImageDraw,
    ImageEnhance,
    ImageFilter,
    ImageFont,
)


MASTER_SIZE = 1024
ICO_SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)
MICRO_MAX_SIZE = 32
APP_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = APP_ROOT / "assets"
SOURCE_PATH = ASSET_DIR / "touch_system_icon_source.png"
PNG_PATH = ASSET_DIR / "touch_system_icon.png"
ICO_PATH = ASSET_DIR / "touch_system_icon.ico"
PREVIEW_PATH = ASSET_DIR / "touch_system_icon_preview.png"


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    filename = "segoeuib.ttf" if bold else "segoeui.ttf"
    path = Path("C:/Windows/Fonts") / filename
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _vertical_gradient(
    size: int,
    top: tuple[int, int, int],
    bottom: tuple[int, int, int],
    mask: Image.Image,
) -> Image.Image:
    data = np.zeros((size, size, 4), dtype=np.uint8)
    for y in range(size):
        ratio = y / max(size - 1, 1)
        for channel in range(3):
            data[y, :, channel] = round(
                top[channel] * (1.0 - ratio) + bottom[channel] * ratio
            )
        data[y, :, 3] = 255
    image = Image.fromarray(data, "RGBA")
    image.putalpha(mask)
    return image


def load_master_source() -> Image.Image:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"Missing icon source: {SOURCE_PATH}. Keep the approved transparent "
            "contact-fold artwork in assets before rebuilding."
        )

    source = Image.open(SOURCE_PATH).convert("RGBA")
    bbox = source.getbbox()
    if bbox is None:
        raise ValueError("Icon source is fully transparent")

    subject = source.crop(bbox)
    target = round(MASTER_SIZE * 0.88)
    subject.thumbnail((target, target), Image.Resampling.LANCZOS)
    master = Image.new("RGBA", (MASTER_SIZE, MASTER_SIZE), (0, 0, 0, 0))
    master.alpha_composite(
        subject,
        ((MASTER_SIZE - subject.width) // 2, (MASTER_SIZE - subject.height) // 2),
    )
    return master


def build_large_frame(master: Image.Image, size: int) -> Image.Image:
    frame = master.resize((size, size), Image.Resampling.LANCZOS)
    if size <= 64:
        alpha = frame.getchannel("A")
        rgb = ImageEnhance.Color(frame.convert("RGB")).enhance(1.10)
        rgb = ImageEnhance.Contrast(rgb).enhance(1.05)
        rgb.putalpha(alpha)
        frame = rgb.filter(ImageFilter.UnsharpMask(radius=0.55, percent=75, threshold=2))
    return frame


def build_micro_frame(size: int) -> Image.Image:
    """Draw size-specific artwork so the open notch survives at 16-32 px."""

    scale = 20
    canvas_size = size * scale

    base = Image.new("L", (canvas_size, canvas_size), 0)
    base_draw = ImageDraw.Draw(base)
    x0, y0, x1, y1 = (
        round(0.06 * canvas_size),
        round(0.075 * canvas_size),
        round(0.79 * canvas_size),
        round(0.925 * canvas_size),
    )
    base_draw.rounded_rectangle(
        (x0, y0, x1, y1),
        radius=round(0.20 * canvas_size),
        fill=255,
    )

    pocket_x = round(0.765 * canvas_size)
    pocket_y = round(0.465 * canvas_size)
    inner_radius = round(0.255 * canvas_size)
    outer_radius = round(0.315 * canvas_size)

    inner = Image.new("L", (canvas_size, canvas_size), 0)
    ImageDraw.Draw(inner).ellipse(
        (
            pocket_x - inner_radius,
            pocket_y - inner_radius,
            pocket_x + inner_radius,
            pocket_y + inner_radius,
        ),
        fill=255,
    )
    outer = Image.new("L", (canvas_size, canvas_size), 0)
    ImageDraw.Draw(outer).ellipse(
        (
            pocket_x - outer_radius,
            pocket_y - outer_radius,
            pocket_x + outer_radius,
            pocket_y + outer_radius,
        ),
        fill=255,
    )

    body = ImageChops.subtract(base, inner)
    teal_ring = ImageChops.multiply(base, ImageChops.subtract(outer, inner))
    frame = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))

    frame.alpha_composite(
        _vertical_gradient(canvas_size, (18, 170, 201), (8, 132, 169), body)
    )
    edge_pixels = max(1, round(0.018 * canvas_size))
    inset_body = body.filter(ImageFilter.MinFilter(edge_pixels * 2 + 1))
    frame.alpha_composite(
        _vertical_gradient(canvas_size, (91, 222, 241), (39, 187, 220), inset_body)
    )
    frame.alpha_composite(
        _vertical_gradient(canvas_size, (0, 111, 140), (0, 62, 86), teal_ring)
    )

    sphere_x = round(0.84 * canvas_size)
    sphere_y = round(0.465 * canvas_size)
    sphere_radius = round(0.205 * canvas_size)
    sphere = Image.new("L", (canvas_size, canvas_size), 0)
    ImageDraw.Draw(sphere).ellipse(
        (
            sphere_x - sphere_radius,
            sphere_y - sphere_radius,
            sphere_x + sphere_radius,
            sphere_y + sphere_radius,
        ),
        fill=255,
    )
    frame.alpha_composite(
        _vertical_gradient(canvas_size, (218, 91, 76), (167, 55, 48), sphere)
    )
    inset_sphere = sphere.filter(ImageFilter.MinFilter(edge_pixels * 2 + 1))
    frame.alpha_composite(
        _vertical_gradient(canvas_size, (255, 159, 137), (244, 93, 77), inset_sphere)
    )

    return frame.resize((size, size), Image.Resampling.LANCZOS)


def build_frames(master: Image.Image) -> dict[int, Image.Image]:
    return {
        size: (
            build_micro_frame(size)
            if size <= MICRO_MAX_SIZE
            else build_large_frame(master, size)
        )
        for size in ICO_SIZES
    }


def save_icon_files(master: Image.Image, frames: dict[int, Image.Image]) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    master.save(PNG_PATH, optimize=True)

    base = frames[256]
    append_images = [frames[size] for size in ICO_SIZES if size != 256]
    base.save(
        ICO_PATH,
        format="ICO",
        sizes=[(size, size) for size in ICO_SIZES],
        append_images=append_images,
    )


def _paste_centered(
    canvas: Image.Image,
    icon: Image.Image,
    box: tuple[int, int, int, int],
) -> None:
    left, top, right, bottom = box
    x = left + (right - left - icon.width) // 2
    y = top + (bottom - top - icon.height) // 2
    canvas.alpha_composite(icon, (x, y))


def build_preview(master: Image.Image, frames: dict[int, Image.Image]) -> None:
    preview = Image.new("RGBA", (1500, 860), "#EEF4F8")
    draw = ImageDraw.Draw(preview)
    draw.text(
        (42, 28),
        "Contact-fold app icon - multi-resolution review",
        font=_font(30, bold=True),
        fill="#102236",
    )

    draw.rounded_rectangle(
        (42, 92, 548, 812), radius=30, fill="#FFFFFF", outline="#C6D9E7", width=2
    )
    draw.text((72, 122), "Master artwork", font=_font(22, bold=True), fill="#102236")
    large = master.resize((420, 420), Image.Resampling.LANCZOS)
    _paste_centered(preview, large, (82, 190, 508, 616))
    draw.text(
        (72, 735),
        "Soft 3D at 48 px and above",
        font=_font(18),
        fill="#607487",
    )

    sizes = (128, 64, 48, 32, 24, 16)
    row_specs = (
        ("LIGHT", "#FFFFFF", 92),
        ("DARK", "#142638", 452),
    )
    for label, background, row_top in row_specs:
        draw.text(
            (592, row_top + 12),
            label,
            font=_font(18, bold=True),
            fill="#102236",
        )
        x = 592
        for size in sizes:
            cell = (x, row_top + 52, x + 138, row_top + 328)
            draw.rounded_rectangle(
                cell,
                radius=18,
                fill=background,
                outline="#BCD1E1" if label == "LIGHT" else "#36536B",
                width=2,
            )
            _paste_centered(preview, frames[size], (x, row_top + 72, x + 138, row_top + 242))
            draw.text(
                (x + 14, row_top + 284),
                f"{size} px",
                font=_font(16),
                fill="#607487" if label == "LIGHT" else "#C5D9E7",
            )
            x += 148

    preview.convert("RGB").save(PREVIEW_PATH, quality=96)


def validate_assets(master: Image.Image) -> None:
    if master.getpixel((0, 0))[3] != 0:
        raise ValueError("Master icon corners must remain transparent")

    bbox = master.getbbox()
    if bbox is None or min(bbox[0], bbox[1], MASTER_SIZE - bbox[2], MASTER_SIZE - bbox[3]) < 35:
        raise ValueError(f"Master icon has insufficient safe margin: {bbox}")

    icon = Image.open(ICO_PATH)
    actual_sizes = {width for width, height in icon.ico.sizes() if width == height}
    missing = set(ICO_SIZES) - actual_sizes
    if missing:
        raise ValueError(f"ICO is missing size-specific frames: {sorted(missing)}")

    micro = icon.ico.getimage((16, 16)).convert("RGBA")
    pixels = np.asarray(micro)
    alpha = pixels[:, :, 3] > 32
    cyan = alpha & (pixels[:, :, 2] > 135) & (pixels[:, :, 1] > 115) & (pixels[:, :, 0] < 125)
    teal = alpha & (pixels[:, :, 0] < 45) & (pixels[:, :, 1] < 150) & (pixels[:, :, 2] < 155)
    coral = alpha & (pixels[:, :, 0] > 180) & (pixels[:, :, 1] < 180) & (pixels[:, :, 2] < 170)
    if alpha.sum() < 45 or cyan.sum() < 20 or teal.sum() < 3 or coral.sum() < 3:
        raise ValueError(
            "16 px artwork lost its three-color recognition signature: "
            f"visible={alpha.sum()}, cyan={cyan.sum()}, teal={teal.sum()}, coral={coral.sum()}"
        )


def main() -> None:
    master = load_master_source()
    frames = build_frames(master)
    save_icon_files(master, frames)
    build_preview(master, frames)
    validate_assets(master)
    print(f"PNG: {PNG_PATH}")
    print(f"ICO: {ICO_PATH}")
    print(f"Preview: {PREVIEW_PATH}")


if __name__ == "__main__":
    main()
