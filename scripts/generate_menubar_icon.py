"""Regenerate the menu bar app's animated bust icon frames.

Draws a bronze/gold medallion bust of Ulysses (Odysseus), identifiable by
his *pilos* travel cap, in the style of ancient Greek/Roman coin portraiture.
24 frames sweep a highlight around the disc to give the menu bar icon a
subtle rotating-light animation.

The face/cap/beard/ear outlines are defined as landmark points and turned
into smooth SVG paths via Catmull-Rom-to-Bezier interpolation (`_spline.py`)
rather than hand-authored bezier control points -- hand-guessing bezier
handles for organic/anatomical shapes produced unrecognizable blob shapes;
landmark points + spline interpolation nailed a recognizable profile on the
first attempt.

Usage: uv run --with pillow python3 scripts/generate_menubar_icon.py
Requires macOS's built-in `qlmanage` to rasterize SVG -> PNG.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path

from _spline import catmull_rom_to_bezier

SIZE = 512
DISPLAY_SIZE = 44
N_FRAMES = 24
DEST_DIR = Path(__file__).resolve().parent.parent / "ulysses" / "app" / "assets" / "icon_frames"

HEAD_POINTS = [
    (270, 60),
    (310, 90),
    (345, 150),
    (365, 190),
    (385, 230),
    (350, 248),
    (360, 262),
    (350, 270),
    (358, 282),
    (348, 305),
    (320, 322),
    (290, 335),
    (280, 400),
    (280, 430),
    (200, 430),
    (205, 360),
    (185, 280),
    (168, 200),
    (185, 120),
    (230, 75),
]
CAP_POINTS = [
    (330, 145),
    (340, 95),
    (300, 55),
    (250, 40),
    (200, 50),
    (160, 90),
    (150, 150),
    (160, 205),
    (185, 225),
    (220, 190),
    (280, 165),
]
BEARD_POINTS = [
    (208, 222),
    (222, 258),
    (255, 288),
    (295, 312),
    (318, 330),
    (304, 352),
    (276, 362),
    (244, 352),
    (216, 322),
    (200, 282),
    (198, 240),
]
EAR_POINTS = [
    (206, 218),
    (222, 212),
    (232, 226),
    (230, 246),
    (216, 250),
    (204, 238),
]


def path(points: list[tuple[float, float]], closed: bool = True) -> str:
    return catmull_rom_to_bezier(points, closed=closed)


def build_svg(highlight_angle_deg: float) -> str:
    a = math.radians(highlight_angle_deg)
    x1, y1 = 256 + 230 * math.cos(a), 256 + 230 * math.sin(a)
    x2, y2 = 256 - 230 * math.cos(a), 256 - 230 * math.sin(a)

    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{SIZE}" height="{SIZE}" viewBox="0 0 {SIZE} {SIZE}">
  <defs>
    <linearGradient id="skin" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#f2d9a0"/>
      <stop offset="30%" stop-color="#d6a24f"/>
      <stop offset="60%" stop-color="#8f5a28"/>
      <stop offset="100%" stop-color="#402611"/>
    </linearGradient>
    <linearGradient id="cap" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#caa062"/>
      <stop offset="40%" stop-color="#7a4d24"/>
      <stop offset="100%" stop-color="#2c1a0c"/>
    </linearGradient>
    <linearGradient id="beard" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#c8994f"/>
      <stop offset="50%" stop-color="#7c4e22"/>
      <stop offset="100%" stop-color="#33200e"/>
    </linearGradient>
    <radialGradient id="disc" cx="35%" cy="30%" r="75%">
      <stop offset="0%" stop-color="#2a1c10"/>
      <stop offset="100%" stop-color="#120b06"/>
    </radialGradient>
  </defs>

  <circle cx="256" cy="256" r="248" fill="url(#disc)"/>
  <circle cx="256" cy="256" r="248" fill="none" stroke="#e8bd76" stroke-width="5" opacity="0.65"/>
  <circle cx="256" cy="256" r="234" fill="none" stroke="#e8bd76" stroke-width="1.5" opacity="0.35"/>

  <path d="{path(HEAD_POINTS)}" fill="url(#skin)" stroke="#1c1006" stroke-width="4"/>
  <path d="{path(BEARD_POINTS)}" fill="url(#beard)" stroke="#1c1006" stroke-width="3"/>
  <path d="{path(EAR_POINTS)}" fill="url(#skin)" stroke="#1c1006" stroke-width="2.5"/>
  <path d="{path(CAP_POINTS)}" fill="url(#cap)" stroke="#1c1006" stroke-width="4"/>

  <path d="M 298,138 C 313,132 329,133 342,143" stroke="#1c1006" stroke-width="4" fill="none" stroke-linecap="round"/>
  <path d="M 313,170 C 321,164 334,164 343,171 C 334,177 321,177 313,170 Z" fill="#1c1006"/>
  <circle cx="330" cy="169" r="2.2" fill="#e8bd76"/>
  <path d="M 300,190 C 310,205 314,222 308,238" stroke="#1c1006" stroke-width="2" fill="none" opacity="0.55" stroke-linecap="round"/>
  <path d="M 232,345 C 245,368 252,392 250,415" stroke="#1c1006" stroke-width="2" fill="none" opacity="0.4" stroke-linecap="round"/>
</svg>'''


def main() -> None:
    from PIL import Image

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for i in range(N_FRAMES):
            angle = 360 * i / N_FRAMES
            svg_path = tmp_path / f"frame_{i:02d}.svg"
            svg_path.write_text(build_svg(angle))
            subprocess.run(
                ["qlmanage", "-t", "-s", "256", "-o", str(tmp_path), str(svg_path)],
                check=True,
                capture_output=True,
            )

        for i in range(N_FRAMES):
            rendered = tmp_path / f"frame_{i:02d}.svg.png"
            img = (
                Image.open(rendered)
                .convert("RGBA")
                .resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS)
            )
            img.save(DEST_DIR / f"frame_{i:02d}.png", optimize=True)

    print(f"wrote {N_FRAMES} frames ({DISPLAY_SIZE}x{DISPLAY_SIZE}) to {DEST_DIR}")


if __name__ == "__main__":
    main()
