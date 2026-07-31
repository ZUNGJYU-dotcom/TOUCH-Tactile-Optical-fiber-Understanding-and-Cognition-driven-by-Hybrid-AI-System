from __future__ import annotations

from math import cos, pi, sin
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


SIZE = 1024
SCALE = 4
PHI = (1 + 5**0.5) / 2
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "assets" / "icon_concepts"
ICON_PATH = OUTPUT_DIR / "concept_d_minimal_touch_t.png"
PREVIEW_PATH = OUTPUT_DIR / "concept_d_minimal_touch_t_sizes.png"


def superellipse(cx: float, cy: float, rx: float, ry: float, power: float = 5.0) -> list[tuple[float, float]]:
    exponent = 2.0 / power
    points: list[tuple[float, float]] = []
    for index in range(720):
        angle = 2 * pi * index / 720
        x_value = cos(angle)
        y_value = sin(angle)
        points.append(
            (
                cx + rx * (1 if x_value >= 0 else -1) * abs(x_value) ** exponent,
                cy + ry * (1 if y_value >= 0 else -1) * abs(y_value) ** exponent,
            )
        )
    return points


def scale_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [(x * SCALE, y * SCALE) for x, y in points]


def gradient(top: tuple[int, int, int], bottom: tuple[int, int, int], size: int) -> Image.Image:
    strip = Image.new("RGBA", (1, size))
    strip.putdata(
        [
            (
                *(round(top[channel] * (1 - ratio) + bottom[channel] * ratio) for channel in range(3)),
                255,
            )
            for ratio in (index / (size - 1) for index in range(size))
        ]
    )
    return strip.resize((size, size))


def build_icon() -> Image.Image:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    canvas_size = SIZE * SCALE

    outer_mask = Image.new("L", (canvas_size, canvas_size), 0)
    ImageDraw.Draw(outer_mask).polygon(scale_points(superellipse(512, 512, 448, 448)), fill=255)

    icon = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    icon.paste(gradient((250, 253, 254), (227, 241, 247), canvas_size), (0, 0), outer_mask)

    glyph_mask = Image.new("L", (canvas_size, canvas_size), 0)
    glyph_draw = ImageDraw.Draw(glyph_mask)
    crossbar_width = SIZE / PHI
    crossbar_left = (SIZE - crossbar_width) / 2
    crossbar_right = SIZE - crossbar_left
    glyph_draw.rounded_rectangle(
        (
            crossbar_left * SCALE,
            270 * SCALE,
            crossbar_right * SCALE,
            430 * SCALE,
        ),
        radius=80 * SCALE,
        fill=255,
    )
    glyph_draw.polygon(
        scale_points([(440, 360), (584, 360), (570, 700), (454, 700)]),
        fill=255,
    )
    glyph_draw.ellipse(
        (454 * SCALE, 642 * SCALE, 570 * SCALE, 758 * SCALE),
        fill=255,
    )
    glyph_mask = glyph_mask.filter(ImageFilter.GaussianBlur(1.2 * SCALE))

    shadow = Image.new("RGBA", (canvas_size, canvas_size), (12, 69, 91, 0))
    shifted = glyph_mask.transform(
        glyph_mask.size,
        Image.Transform.AFFINE,
        (1, 0, 0, 0, 1, -14 * SCALE),
        resample=Image.Resampling.BICUBIC,
    ).filter(ImageFilter.GaussianBlur(18 * SCALE))
    shadow.putalpha(shifted.point(lambda value: round(value * 0.22)))
    icon.alpha_composite(shadow)

    glyph = gradient((30, 183, 207), (8, 115, 150), canvas_size)
    glyph.putalpha(glyph_mask)
    icon.alpha_composite(glyph)

    highlight_mask = Image.new("L", (canvas_size, canvas_size), 0)
    highlight_draw = ImageDraw.Draw(highlight_mask)
    highlight_draw.polygon(
        scale_points([(170, 230), (590, 230), (502, 758), (270, 500)]),
        fill=215,
    )
    highlight_mask = Image.composite(highlight_mask, Image.new("L", highlight_mask.size, 0), glyph_mask)
    highlight = Image.new("RGBA", (canvas_size, canvas_size), (151, 229, 237, 0))
    highlight.putalpha(highlight_mask.filter(ImageFilter.GaussianBlur(3 * SCALE)))
    icon.alpha_composite(highlight)

    outline = ImageDraw.Draw(icon)
    outline.line(
        scale_points(superellipse(512, 512, 438, 438)),
        fill=(255, 255, 255, 115),
        width=4 * SCALE,
    )

    result = icon.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    result.save(ICON_PATH, optimize=True)
    return result


def build_preview(icon: Image.Image) -> None:
    preview = Image.new("RGB", (1120, 360), "#EEF3F6")
    draw = ImageDraw.Draw(preview)
    title_font = ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", 27)
    label_font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 22)
    draw.text((40, 28), "TOUCH custom T - recognition at operating sizes", fill="#102236", font=title_font)

    x_position = 46
    for icon_size in (192, 128, 64, 32, 16):
        tile_size = max(208, icon_size + 28)
        y_position = 105 + (192 - icon_size) // 2
        draw.rounded_rectangle(
            (x_position, 92, x_position + tile_size, 318),
            radius=22,
            fill="white",
            outline="#D8E3EC",
            width=2,
        )
        sample = icon.resize((icon_size, icon_size), Image.Resampling.LANCZOS)
        preview.paste(sample, (x_position + (tile_size - icon_size) // 2, y_position), sample)
        draw.text((x_position + 14, 286), f"{icon_size} px", fill="#607487", font=label_font)
        x_position += tile_size + 18
    preview.save(PREVIEW_PATH, optimize=True)


if __name__ == "__main__":
    built_icon = build_icon()
    build_preview(built_icon)
