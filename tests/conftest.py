"""測試共用：把專案根目錄加進 sys.path，並提供 stub 偵測器與合成影片工具。"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import blur_faces as bf  # noqa: E402

import cv2  # noqa: E402


class StubDetector(bf._ScoredDetector):
    """固定回傳預先排好的偵測結果；每呼叫一次 detect_scored 前進一幀。"""

    provider = "CPUExecutionProvider"
    det_size = 1920

    def __init__(self, per_frame=None, conf: float = 0.5):
        self.conf = conf
        self.per_frame = per_frame or []
        self.calls = 0

    def detect_scored(self, frame):
        i = self.calls
        self.calls += 1
        if callable(self.per_frame):
            return list(self.per_frame(i, frame))
        return list(self.per_frame[i]) if i < len(self.per_frame) else []


class BrokenDetector(bf._ScoredDetector):
    """模擬偵測器在 GPU 上出錯（裝置遺失）：第 fail_at 幀之後每次都丟例外。"""

    provider = "CPUExecutionProvider"
    det_size = 640

    def __init__(self, fail_at: int = 3):
        self.conf = 0.5
        self.calls = 0
        self.fail_at = fail_at

    def detect_scored(self, frame):
        self.calls += 1
        if self.calls > self.fail_at:
            raise RuntimeError("D3D12 device removed")
        return []


def make_video(path: Path, frames: int = 60, size=(640, 360), fps: int = 30, audio: bool = False) -> Path:
    """合成一支移動圓形的影片。audio=True 時用 ffmpeg 加一條 AAC 正弦波音軌（需要 ffmpeg）。"""
    raw = path if not audio else path.with_name(path.stem + "_noaudio.mp4")
    wr = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(frames):
        f = np.full((size[1], size[0], 3), 200, np.uint8)
        cv2.circle(f, (size[0] // 4 + i * 2, size[1] // 2), 50, (30, 60, 200), -1)
        wr.write(f)
    wr.release()
    if audio:
        import subprocess

        ffmpeg = bf.find_ffmpeg()
        assert ffmpeg, "測試需要 ffmpeg"
        subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(raw),
             "-f", "lavfi", "-i", f"sine=frequency=440:duration={frames / fps:.3f}",
             "-c:v", "copy", "-c:a", "aac", "-shortest", str(path)],
            check=True, **bf._subprocess_kwargs(),
        )
        raw.unlink()
    return path


def video_args(**over):
    base = dict(mode="mosaic", strength=5, pad=0.15, keep=4, ellipse=False, track=True,
                min_hits=2, smooth=6, encoder="software")
    base.update(over)
    import argparse

    return argparse.Namespace(**base)


@pytest.fixture
def tmp_zh(tmp_path):
    """含中文與空白的暫存目錄：Windows 上 OpenCV / ffmpeg 的路徑處理都要吃得下。"""
    d = tmp_path / "測試 目錄"
    d.mkdir()
    return d
