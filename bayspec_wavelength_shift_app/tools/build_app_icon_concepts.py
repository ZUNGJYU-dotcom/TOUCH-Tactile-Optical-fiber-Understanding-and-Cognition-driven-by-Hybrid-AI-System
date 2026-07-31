from __future__ import annotations

from math import cos, pi, sin
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


SIZE = 1024
SCALE = 3
PHI = (1 + 5**0.5) / 2
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "assets" / "icon_concepts"


def superellipse(cx: float, cy: float, rx: float, ry: float, power: float = 5.0) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    exponent = 2.0 / power
    for index in range(720):
        angle = 2 * pi * index / 720
        x_value = cos(angle)
        y_value = sin(angle)
        result.append(
            (
                cx + rx * (1 if x_value >= 0 else -1) * abs(x_value) ** exponent,
                cy + ry * (1 if y_value >= 0 else -1) * abs(y_value) ** exponent,
            )
        )
    return result


def scale_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [(x * SCALE, y * SCALE) for x, y in points]


def box(values: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return tuple(value * SCALE for value in values)


def gradient(top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    canvas_size = SIZE * SCALE
    strip = Image.new("RGBA", (1, canvas_size))
    colors = []
    for index in range(canvas_size):
        ratio = index / (canvas_size - 1)
        colors.append(
            (
                *(round(top[channel] * (1 - ratio) + bottom[channel] * ratio) for channel in range(3)),
                255,
            )
        )
    strip.putdata(colors)
    return strip.resize((canvas_size, canvas_size))


def base_icon(top: tuple[int, int, int], bottom: tuple[int, int, int]) -> Image.Image:
    canvas_size = SIZE * SCALE
    mask = Image.new("L", (canvas_size, canvas_size), 0)
    ImageDraw.Draw(mask).polygon(scale_points(superellipse(512, 512, 448, 448)), fill=255)
    icon = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    icon.paste(gradient(top, bottom), (0, 0), mask)
    return icon


def finish(icon: Image.Image, name: str) -> Image.Image:
    draw = ImageDraw.Draw(icon)
    draw.line(
        scale_points(superellipse(512, 512, 437, 437)),
        fill=(255, 255, 255, 92),
        width=4 * SCALE,
    )
    result = icon.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    result.save(OUTPUT_DIR / f"{name}.png", optimize=True)
    return result


def concept_a_contact_ripple() -> Image.Image:
    icon = base_icon((20, 56, 77), (7, 142, 172))
    draw = ImageDraw.Draw(icon)

    center_x = 512 * SCALE
    dot_y = 345 * SCALE
    dot_radius = 56 * SCALE
    draw.ellipse(
        (center_x - dot_radius, dot_y - dot_radius, center_x + dot_radius, dot_y + dot_radius),
        fill="#F6FCFE",
    )

    widths = (SIZE / PHI**2, SIZE / PHI, SIZE / PHI * 1.18)
    heights = (190, 300, 410)
    strokes = (38, 34, 30)
    for width, height, stroke in zip(widths, heights, strokes):
        bounds = box((512 - width / 2, 345, 512 + width / 2, 345 + height))
        draw.arc(bounds, start=24, end=156, fill="#F6FCFE", width=stroke * SCALE)

    glow = Image.new("RGBA", icon.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_draw.ellipse(box((454, 287, 570, 403)), fill=(116, 231, 241, 72))
    icon.alpha_composite(glow.filter(ImageFilter.GaussianBlur(30 * SCALE)))
    draw = ImageDraw.Draw(icon)
    draw.ellipse(box((474, 307, 550, 383)), fill="#F6FCFE")
    return finish(icon, "concept_a_contact_ripple")


def concept_b_optical_pad() -> Image.Image:
    icon = base_icon((249, 252, 253), (218, 239, 246))
    canvas_size = SIZE * SCALE

    shadow = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).ellipse(box((206, 343, 818, 721)), fill=(23, 72, 91, 45))
    icon.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(30 * SCALE)))

    pad = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    pad_draw = ImageDraw.Draw(pad)
    pad_draw.ellipse(box((206, 321, 818, 699)), fill="#1AA7C3")
    pad_draw.ellipse(box((229, 344, 795, 676)), fill="#5AC9D8")
    pad_draw.ellipse(box((372, 393, 652, 673)), fill="#1689A8")
    pad_draw.ellipse(box((425, 446, 599, 620)), fill="#15374A")
    pad_draw.ellipse(box((461, 482, 563, 584)), fill="#E9F9FC")
    pad = pad.filter(ImageFilter.GaussianBlur(1.2 * SCALE))
    icon.alpha_composite(pad)
    return finish(icon, "concept_b_optical_pad")


def concept_c_touch_monogram() -> Image.Image:
    icon = base_icon((11, 124, 158), (26, 190, 207))
    draw = ImageDraw.Draw(icon)

    glyph_width = SIZE / PHI
    left = (SIZE - glyph_width) / 2
    right = left + glyph_width
    bar_y = 382
    draw.line(
        scale_points([(left, bar_y), (right, bar_y)]),
        fill="#F7FCFD",
        width=92 * SCALE,
    )
    draw.line(
        scale_points([(512, bar_y), (512, 682)]),
        fill="#F7FCFD",
        width=92 * SCALE,
    )
    draw.ellipse(box((432, 626, 592, 786)), fill="#17364A")
    draw.ellipse(box((474, 668, 550, 744)), fill="#8CE3EA")
    return finish(icon, "concept_c_touch_monogram")


def make_contact_sheet(icons: list[tuple[str, Image.Image]]) -> None:
    sheet = Image.new("RGB", (1680, 720), "#F4F7FA")
    draw = ImageDraw.Draw(sheet)
    font_path = Path("C:/Windows/Fonts/segoeuib.ttf")
    regular_path = Path("C:/Windows/Fonts/segoeui.ttf")
    title_font = ImageFont.truetype(str(font_path), 34)
    label_font = ImageFont.truetype(str(font_path), 25)
    note_font = ImageFont.truetype(str(regular_path), 20)

    draw.text((60, 38), "TOUCH System - app icon directions", fill="#102236", font=title_font)
    draw.text((60, 86), "Golden-ratio safe zone | one core symbol | tested at shortcut size", fill="#607487", font=note_font)

    card_width = 500
    for index, (label, icon) in enumerate(icons):
        x = 60 + index * 540
        draw.rounded_rectangle((x, 136, x + card_width, 660), radius=28, fill="white", outline="#D8E3EC", width=2)
        preview = icon.resize((390, 390), Image.Resampling.LANCZOS)
        sheet.paste(preview, (x + 55, 170), preview)
        draw.text((x + 36, 574), label, fill="#102236", font=label_font)
        small = icon.resize((56, 56), Image.Resampling.LANCZOS)
        sheet.paste(small, (x + 405, 570), small)

    sheet.save(OUTPUT_DIR / "touch_system_icon_concepts.png", optimize=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    icons = [
        ("A  Contact ripple", concept_a_contact_ripple()),
        ("B  Optical pad", concept_b_optical_pad()),
        ("C  TOUCH monogram", concept_c_touch_monogram()),
    ]
    make_contact_sheet(icons)


if __name__ == "__main__":
    main()
