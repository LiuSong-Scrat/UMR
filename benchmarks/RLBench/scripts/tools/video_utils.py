#!/usr/bin/env python3
"""Shared video overlays for RLBench evaluation and replay tools."""

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def annotate_final_task_result(frame, success):
    """Draw the final episode result in the upper-right corner of one frame."""
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(image)
    padding = max(5, int(round(image.width / 100.0)))
    font_size = max(12, min(28, int(round(image.width / 22.0))))
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        font = ImageFont.load_default()

    label = "SUCCESS: TRUE" if success else "SUCCESS: FALSE"
    color = (45, 225, 80) if success else (255, 55, 45)
    text_box = draw.textbbox((0, 0), label, font=font, stroke_width=1)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    panel_width = text_width + 2 * padding
    panel_height = text_height + 2 * padding
    panel_x = max(0, image.width - panel_width - padding)
    panel_y = padding
    draw.rectangle(
        (panel_x, panel_y, panel_x + panel_width, panel_y + panel_height),
        fill=(0, 0, 0),
        outline=color,
        width=max(2, padding // 2),
    )
    draw.text(
        (panel_x + padding, panel_y + padding - text_box[1]),
        label,
        fill=color,
        font=font,
        stroke_width=1,
        stroke_fill=(0, 0, 0),
    )
    return np.asarray(image, dtype=np.uint8)


def annotate_final_task_result_frames(frames, success):
    """Apply the final result badge to every frame in an episode video."""
    return [annotate_final_task_result(frame, success) for frame in frames]
