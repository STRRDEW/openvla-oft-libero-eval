#!/usr/bin/env python3
"""Build the 60-second project demo from real LIBERO rollout clips.

All robot footage is from the actual evaluation runs; the only generated
frames are the title/method/summary cards and text overlays. Clips are played
at 1x (no acceleration, no slow motion); the video says so on screen.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
FPS = 30
BG = (14, 16, 20)
PANEL = (24, 27, 33)
WHITE = (240, 242, 245)
MUTED = (158, 166, 178)
GREEN = (52, 199, 123)
RED = (235, 87, 87)
ACCENT = (88, 166, 255)
YELLOW = (245, 190, 80)

FONT_REG = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)


def text(draw, xy, s, f, fill=WHITE, anchor=None):
    draw.text(xy, s, font=f, fill=fill, anchor=anchor)


def wrap(draw, s: str, f, max_width: int) -> list[str]:
    lines, line = [], ""
    for word in s.split(" "):
        candidate = (line + " " + word).strip()
        if draw.textlength(candidate, font=f) <= max_width:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def base_canvas() -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, 6), fill=ACCENT)
    d.text((48, 28), "OpenVLA-OFT  +  LIBERO-Spatial", font=font(24, True), fill=WHITE)
    d.text((W - 48, 32), "真实 rollout · 30 fps", font=font(18), fill=MUTED, anchor="ra")
    d.line((48, 72, W - 48, 72), fill=(48, 53, 62), width=1)
    return img


def card(kicker: str, title: str, bullets: list[str], duration: float, accent=ACCENT):
    frames = int(duration * FPS)
    for i in range(frames):
        img = base_canvas()
        d = ImageDraw.Draw(img)
        d.text((48, 120), kicker, font=font(18, True), fill=accent)
        d.text((48, 154), title, font=font(44, True), fill=WHITE)
        y = 260
        for b in bullets:
            d.ellipse((50, y + 12, 60, y + 22), fill=accent)
            for ln in wrap(d, b, font(24), W - 160):
                d.text((78, y), ln, font=font(24), fill=WHITE)
                y += 38
            y += 14
        d.text((48, H - 54), "真实实验录像  ·  未使用生成视频替代", font=font(18), fill=MUTED)
        yield np.array(img)


def clip_frame(frame_bgr, task, config, success, metrics, idx, total, t, duration):
    img = base_canvas()
    d = ImageDraw.Draw(img)
    # left: real frame
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA)
    img.paste(Image.fromarray(rgb), (60, 104))
    d.rectangle((60, 104, 571, 615), outline=(90, 98, 112), width=2)
    d.text((60, 628), f"CLIP {idx}/{total}   ·   t = {t:4.1f}s", font=font(16), fill=MUTED)

    # right: task / config / result / metrics
    x = 640
    d.text((x, 100), "TASK", font=font(14, True), fill=ACCENT)
    y = 124
    for ln in wrap(d, task, font(24, True), 580)[:4]:
        d.text((x, y), ln, font=font(24, True), fill=WHITE)
        y += 31
    y += 8
    d.text((x, y), config, font=font(17), fill=MUTED)
    y += 42
    color = GREEN if success else RED
    label = "SUCCESS" if success else "FAILURE / TIMEOUT"
    d.rounded_rectangle((x, y, x + 210, y + 40), radius=6, fill=(30, 52, 42) if success else (58, 30, 32),
                        outline=color, width=1)
    d.text((x + 14, y + 8), label, font=font(18, True), fill=color)
    y += 66
    for k, v in metrics:
        d.text((x, y), k, font=font(16), fill=MUTED)
        d.text((x + 230, y), v, font=font(17, True), fill=WHITE)
        y += 36
    d.text((x, H - 58), "片段 1× 实速播放，无加速", font=font(16), fill=YELLOW)
    # progress bar
    d.rectangle((60, 680, 1220, 688), fill=(45, 50, 60))
    d.rectangle((60, 680, 60 + int(1160 * (idx - 1 + t / max(duration, 1e-6)) / total), 688), fill=ACCENT)
    return np.array(img)


def play_clip(path, task, config, success, metrics, idx, total, speed=1.0):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        raise RuntimeError(f"empty video: {path}")
    for i, f in enumerate(frames):
        yield clip_frame(f, task, config, success, metrics, idx, total, i / FPS, len(frames) / FPS)


def freeze_frame(path, task, config, success, metrics, idx, total, note, duration=3.0):
    frame = cv2.imread(path)
    if frame is None:
        raise RuntimeError(f"missing frame: {path}")
    for i in range(int(duration * FPS)):
        img = base_canvas()
        d = ImageDraw.Draw(img)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA)
        img.paste(Image.fromarray(rgb), (60, 104))
        d.rectangle((60, 104, 571, 615), outline=(90, 98, 112), width=2)
        x = 640
        d.text((x, 100), "KEY FRAME", font=font(14, True), fill=YELLOW)
        y = 124
        for ln in wrap(d, task, font(22, True), 580)[:4]:
            d.text((x, y), ln, font=font(22, True), fill=WHITE)
            y += 29
        y += 8
        d.text((x, y), config, font=font(16), fill=MUTED)
        y += 44
        color = GREEN if success else RED
        d.text((x, y), "SUCCESS" if success else "FAILURE / TIMEOUT", font=font(18, True), fill=color)
        y += 48
        for k, v in metrics:
            d.text((x, y), k, font=font(16), fill=MUTED)
            d.text((x + 230, y), v, font=font(17, True), fill=WHITE)
            y += 34
        d.text((x, y + 10), note, font=font(18, True), fill=YELLOW)
        d.text((x, H - 58), "关键帧来自真实录像，非生成画面", font=font(16), fill=MUTED)
        yield np.array(img)


def main():
    out = Path("artifacts/demo_60s.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter failed to open")

    segments = []
    segments.append(card(
        "PROJECT DEMO  ·  2026-10-24",
        "语言条件机械臂闭环评测",
        [
            "OpenVLA-OFT 7B 官方权重 + LIBERO-Spatial",
            "10 个任务 × 10 个 episode；纯推理，无训练",
            "动作块执行长度 k = 1 / 4 / 8 的受控对照",
        ],
        5.0,
    ))
    segments.append(card(
        "实验问题",
        "每次执行多少步预测动作？",
        [
            "模型一次输出 8 个动作（action chunk）",
            "每次执行前 k 个动作，然后重新观察",
            "k=1 最闭环；k=8 执行完整动作块（基线）",
            "固定模型、任务、初始状态、seed 和预处理",
        ],
        6.0,
    ))

    segments.append(play_clip(
        "artifacts/phase2/baseline/success_task0_ep0.mp4",
        "pick up the black bowl between the plate and the ramekin and place it on the plate",
        "k=8  ·  seed=7  ·  task0 ep0",
        True,
        [("policy steps", "79"), ("model calls", "10"), ("inference", "1.78 s"), ("wall", "6.97 s")],
        1, 5,
    ))
    segments.append(play_clip(
        "artifacts/phase2/baseline/success_task8_ep4.mp4",
        "pick up the black bowl next to the plate and place it on the plate",
        "k=8  ·  seed=7  ·  task8 ep4  ·  同一初始状态",
        True,
        [("policy steps", "92"), ("model calls", "12"), ("inference", "1.50 s"), ("wall", "6.82 s")],
        2, 5,
    ))

    # Matched failure: same task, same initial state, same seed; k=4 fails.
    segments.append(play_clip(
        "artifacts/phase3/k4/failure_task8_ep4.mp4",
        "pick up the black bowl next to the plate and place it on the plate",
        "k=4  ·  seed=7  ·  task8 ep4  ·  同一初始状态",
        False,
        [("policy steps", "220"), ("model calls", "55"), ("inference", "6.88 s"), ("wall", "15.76 s")],
        3, 5,
    ))
    segments.append(freeze_frame(
        "artifacts/phase3/k4/failure_task8_ep4_end.png",
        "pick up the black bowl next to the plate and place it on the plate",
        "k=4  ·  seed=7  ·  task8 ep4",
        False,
        [("policy steps", "220"), ("model calls", "55"), ("inference", "6.88 s"), ("wall", "15.76 s")],
        3, 5,
        "结束帧：两个碗仍在左侧，夹爪空着停在盘子上方",
    ))

    segments.append(play_clip(
        "artifacts/phase3/k1/failure_task1_ep7.mp4",
        "pick up the black bowl next to the ramekin and place it on the plate",
        "k=1  ·  seed=7  ·  task1 ep7",
        False,
        [("policy steps", "220"), ("model calls", "220"), ("inference", "27.03 s"), ("wall", "35.93 s")],
        4, 5,
    ))
    segments.append(freeze_frame(
        "artifacts/phase3/k1/failure_task1_ep7_end.png",
        "pick up the black bowl next to the ramekin and place it on the plate",
        "k=1  ·  seed=7  ·  task1 ep7",
        False,
        [("policy steps", "220"), ("model calls", "220"), ("inference", "27.03 s"), ("wall", "35.93 s")],
        4, 5,
        "结束帧：目标碗仍在桌面，夹爪停在盘子上方",
    ))

    segments.append(play_clip(
        "artifacts/phase2/baseline/failure_task9_ep2.mp4",
        "pick up the black bowl on the wooden cabinet and place it on the plate",
        "k=8  ·  seed=7  ·  task9 ep2",
        False,
        [("policy steps", "220"), ("model calls", "28"), ("inference", "3.49 s"), ("wall", "12.60 s")],
        5, 5,
    ))

    segments.append(card(
        "结果（10 tasks × 10 episodes）",
        "k=4 与基线同为 99/100；k=1 为 94/100",
        [
            "k=8：99/100，13.32 次模型调用/episode，平均 7.93 s",
            "k=4：99/100，26.51 次调用/episode，平均 9.51 s",
            "k=1：94/100，111.76 次调用/episode，平均 20.28 s",
            "k=1 的 6 个失败全部是 220 次调用后超时；样本小，不宣称显著",
        ],
        8.0,
    ))
    segments.append(card(
        "工程贡献",
        "不是算法创新，而是可复现评测",
        [
            "Blackwell sm_120 环境适配；固定上游 commit",
            "逐 episode JSONL / 同步计时 / 显存 / 视频",
            "发现官方 deque(maxlen=k) 在 k<8 时执行“最后 k 个”",
            "真实视频、失败案例、复现文档",
        ],
        6.0,
    ))
    segments.append(card(
        "谢谢",
        "代码、日志与完整视频见项目 README",
        [
            "OpenVLA-OFT: github.com/moojink/openvla-oft",
            "LIBERO: github.com/Lifelong-Robot-Learning/LIBERO",
            "本视频不含生成式机器人画面",
        ],
        4.0,
    ))

    n = 0
    for seg in segments:
        for frame in seg:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            n += 1
    writer.release()
    print(f"wrote {out}  frames={n}  duration={n/FPS:.2f}s  size={out.stat().st_size}")


if __name__ == "__main__":
    main()
