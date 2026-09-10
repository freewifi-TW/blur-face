"""偵測合併、追蹤器、照片／影片管線與失敗回退的測試。全部用 stub 偵測器與合成影片，不需要 GPU。"""
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

import blur_faces as bf
from conftest import BrokenDetector, StubDetector, make_video, video_args

D = bf.Detection
HAS_FFMPEG = bf.find_ffmpeg() is not None
needs_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="需要 ffmpeg")


# ---------------------------------------------------------------------------
# 偵測框合併
# ---------------------------------------------------------------------------

def test_merge_face_inside_head_always_single_box():
    """臉框在頭框內、IoU 掃過 0.5：永遠合成同一個外接框（舊 NMS 會在臉框／大框間跳）。"""
    widths = set()
    for dy in range(0, 60, 3):
        fb, hb = (500, 400 + dy, 200, 200), (470, 340, 260, 300)
        out = bf._merge_scored([D(fb, 0.9, True), D(hb, 0.8, True)])
        assert len(out) == 1
        ux1, uy1 = min(fb[0], hb[0]), min(fb[1], hb[1])
        ux2, uy2 = max(fb[0] + fb[2], hb[0] + hb[2]), max(fb[1] + fb[3], hb[1] + hb[3])
        assert out[0].box == (ux1, uy1, ux2 - ux1, uy2 - uy1)
        widths.add(out[0].box[2])
    assert widths == {260}


def test_merge_keeps_adjacent_boxes():
    out = bf._merge_scored([D((100, 100, 50, 50), 0.9, True), D((160, 100, 50, 50), 0.9, True)])
    assert len(out) == 2


def test_merge_weak_absorbed_by_strong_keeps_strong_rect():
    out = bf._merge_scored([D((100, 100, 50, 50), 0.2, False), D((105, 100, 50, 50), 0.9, True)])
    assert out == [D((105, 100, 50, 50), 0.9, True)]


def test_union_detector_merges_children():
    face, head = StubDetector(), StubDetector()
    union = bf.UnionDetector([face, head])
    frame = np.zeros((1080, 1920, 3), np.uint8)
    face.per_frame = [[D((500, 400, 200, 200), 0.9, True)]]
    head.per_frame = [[D((470, 340, 260, 300), 0.8, True)]]
    assert union.detect(frame) == [(470, 340, 260, 300)]


# ---------------------------------------------------------------------------
# 追蹤器
# ---------------------------------------------------------------------------

def run_tracker(seq, **kw):
    tr = bf.StreamTracker(**kw)
    out = []
    for dets in seq:
        out.extend(b for _, b in tr.push(np.zeros((2, 2, 3), np.uint8), dets))
    out.extend(b for _, b in tr.flush())
    return out


def test_tracker_smooths_alternating_sizes():
    seq = [[D((500, 400, 200, 200), 0.9, True)] if i % 2 else [D((470, 340, 260, 320), 0.9, True)]
           for i in range(60)]
    out = run_tracker(seq)
    assert len(out) == 60 and all(len(fr) == 1 for fr in out)
    assert {(b[2], b[3]) for fr in out for b in fr} == {(260, 320)}
    assert len({(b[2], b[3]) for fr in run_tracker(seq, smooth=0) for b in fr}) == 2


def test_tracker_weak_boxes_continue_track_past_max_gap():
    seq = ([[D((100, 100, 80, 80), 0.9, True)]] * 5
           + [[D((100 + i, 100, 80, 80), 0.2, False)] for i in range(40)]
           + [[D((140, 100, 80, 80), 0.9, True)]] * 5)
    out = run_tracker(seq)
    assert len(out) == 50 and all(len(fr) == 1 for fr in out)
    # 沒有弱框（舊行為）中段會斷
    assert any(not fr for fr in run_tracker([[d.box for d in fr if d.strong] for fr in seq]))


def test_tracker_weak_boxes_never_start_tracks_or_count_hits():
    assert all(not fr for fr in run_tracker([[D((100, 100, 80, 80), 0.2, False)]] * 30))
    seq = [[D((100, 100, 80, 80), 0.9, True)]] + [[D((100, 100, 80, 80), 0.2, False)]] * 30
    assert all(not fr for fr in run_tracker(seq))  # 只有一次強框 → min_hits 過濾


def test_tracker_plain_boxes_compatible_and_bounded():
    random.seed(1)
    seq, box = [], [300, 300, 120, 120]
    for _ in range(200):
        fr = []
        if random.random() < 0.85:
            box = [box[0] + random.randint(-5, 5), box[1] + random.randint(-5, 5), 120 + random.randint(-10, 10), 120]
            fr.append(tuple(box))
        seq.append(fr)
    out = run_tracker(seq, smooth=0)
    assert len(out) == 200 and sum(1 for fr in out if fr) > 190
    tr = bf.StreamTracker()
    for _ in range(600):
        tr.push(np.zeros((2, 2, 3), np.uint8), [D((100, 100, 80, 80), 0.9, True)])
    assert all(len(t["boxes"]) <= tr.delay + 2 * tr.smooth + 2 for t in tr.tracks)


# ---------------------------------------------------------------------------
# 輸出路徑
# ---------------------------------------------------------------------------

def test_plan_outputs_dedupes_and_maps_heic_to_jpg(tmp_path):
    files = [tmp_path / "a" / "IMG_0001.MOV", tmp_path / "b" / "IMG_0001.mov", tmp_path / "c" / "photo.HEIC"]
    outs = bf.plan_outputs(files, tmp_path / "out")
    assert [o.name for o in outs] == ["IMG_0001_blurred.mp4", "IMG_0001_blurred_2.mp4", "photo_blurred.jpg"]
    assert bf.plan_outputs([files[0]], None) == [tmp_path / "a" / "IMG_0001_blurred.mp4"]


def test_truncated_tolerance():
    assert not bf._truncated(998, 1000)
    assert bf._truncated(990, 1000)
    assert not bf._truncated(0, 0)


# ---------------------------------------------------------------------------
# 照片
# ---------------------------------------------------------------------------

def test_process_image_chinese_path_no_part_left(tmp_zh):
    src = tmp_zh / "照片 中文.png"
    img = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    assert bf._imwrite(src, img)
    out = tmp_zh / "照片 中文_blurred.png"
    det = StubDetector([[D((50, 50, 60, 60), 0.9, True)]])
    n = bf.process_image(src, out, det, video_args(), log=lambda _s: None)
    assert n == 1 and out.exists()
    assert not bf._part_path(out).exists()
    res = bf._imread(out)
    assert res is not None and np.abs(res.astype(int) - img.astype(int))[50:110, 50:110].mean() > 1


def test_process_image_reports_detector_failure(tmp_path):
    src = tmp_path / "x.png"
    bf._imwrite(src, np.zeros((64, 64, 3), np.uint8))
    report = {}
    n = bf.process_image(src, tmp_path / "x_blurred.png", BrokenDetector(fail_at=0), video_args(),
                         log=lambda _s: None, report=report)
    assert n is None and report["detector_failed"] and "偵測失敗" in report["reason"]
    assert not (tmp_path / "x_blurred.png").exists()


@needs_ffmpeg
def test_imread_ffmpeg_fallback_for_unsupported_ext(tmp_path):
    """OpenCV 讀不到的副檔名（HEIC 等）交給 ffmpeg：用 mp4 改名模擬「只有 ffmpeg 能解」的檔。"""
    vid = make_video(tmp_path / "clip.mp4", frames=3)
    fake = tmp_path / "photo.heic"
    fake.write_bytes(vid.read_bytes())
    img = bf._imread(fake)
    assert img is not None and img.shape[:2] == (360, 640)


# ---------------------------------------------------------------------------
# 影片管線
# ---------------------------------------------------------------------------

def bowing_head(i, frame):
    """前 30 幀強框、30-70 幀只剩弱框（低頭）、之後又強框。"""
    box = (140 + i * 2, 120, 120, 120)
    return [D(box, 0.9, True)] if i < 30 or i >= 70 else [D(box, 0.2, False)]


@needs_ffmpeg
def test_process_video_chinese_path_software_encoder(tmp_zh):
    src = make_video(tmp_zh / "測試影片 中文.mp4", frames=90)
    out = tmp_zh / "out_中文.mp4"
    report = {}
    res = bf.process_video(src, out, StubDetector(bowing_head), video_args(), log=lambda _s: None, report=report)
    assert res == (90, 50)
    assert out.exists() and not bf._part_path(out).exists()
    assert report["sink"] == "ffmpeg" and report["encoder"] == "libx264" and report["hw_decode"] is True
    cap = cv2.VideoCapture(str(out))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)  # 弱框區段
    ok, f50 = cap.read()
    cap.release()
    orig = np.full((360, 640, 3), 200, np.uint8)
    cv2.circle(orig, (160 + 50 * 2, 180), 50, (30, 60, 200), -1)
    assert ok and np.abs(f50.astype(int) - orig.astype(int)).mean() > 3


@needs_ffmpeg
def test_process_video_copies_aac_audio(tmp_path):
    src = make_video(tmp_path / "with_audio.mp4", frames=48, audio=True)
    assert bf._probe_audio_codec(bf.find_ffmpeg(), src) == "aac"
    out = tmp_path / "with_audio_blurred.mp4"
    report = {}
    res = bf.process_video(src, out, StubDetector(), video_args(), log=lambda _s: None, report=report)
    assert res is not None and report["audio"] == "copy"
    assert bf._probe_audio_codec(bf.find_ffmpeg(), out) == "aac"


@needs_ffmpeg
def test_process_video_falls_back_from_broken_hw_encoder(tmp_path, monkeypatch):
    """硬體編碼器失敗 → 先改 libx264（保留音軌），並把這個程序後續的 auto 編碼改成軟體。"""
    monkeypatch.setattr(bf, "pick_encoder", lambda mode="auto": ("h264_does_not_exist", []))
    monkeypatch.setitem(bf._encoder_cache, "auto", ("h264_does_not_exist", []))
    src = make_video(tmp_path / "clip.mp4", frames=30, audio=True)
    out = tmp_path / "clip_blurred.mp4"
    logs = []
    report = {}
    res = bf.process_video(src, out, StubDetector(), video_args(encoder="auto"), log=logs.append, report=report)
    assert res == (30, 0) and out.exists()
    assert report["encoder"] == "libx264" and report["sink"] == "ffmpeg"
    assert bf._encoder_cache["auto"] == bf._SOFTWARE_ENCODER
    assert any("改用 libx264" in line for line in logs)
    assert not any(p.suffix == ".mp4" and ".part" in p.name for p in tmp_path.iterdir())


@needs_ffmpeg
def test_process_video_detector_failure_reported_and_cleaned(tmp_path):
    src = make_video(tmp_path / "clip.mp4", frames=30)
    out = tmp_path / "clip_blurred.mp4"
    report = {}
    res = bf.process_video(src, out, BrokenDetector(fail_at=3), video_args(), log=lambda _s: None, report=report)
    assert res is None and report["detector_failed"] and "device removed" in report["reason"]
    assert not out.exists() and not bf._part_path(out).exists()


@needs_ffmpeg
def test_process_video_cancel_cleans_up(tmp_path):
    src = make_video(tmp_path / "clip.mp4", frames=120)
    out = tmp_path / "clip_blurred.mp4"
    seen = {"n": 0}

    def cancel():
        seen["n"] += 1
        return seen["n"] > 20

    report = {}
    res = bf.process_video(src, out, StubDetector(), video_args(), log=lambda _s: None, cancel=cancel, report=report)
    assert res is None and report["reason"] == "已取消"
    assert not out.exists() and not bf._part_path(out).exists()


@needs_ffmpeg
def test_process_video_retries_with_software_decode_when_hw_truncates(tmp_path, monkeypatch):
    """硬體解碼中途無聲停止 → 偵測到幀數不足，改軟體解碼整支重跑。"""
    real_open = bf._open_video

    class EarlyStop:
        def __init__(self, cap, limit):
            self.cap, self.limit, self.n = cap, limit, 0

        def read(self):
            if self.n >= self.limit:
                return False, None
            self.n += 1
            return self.cap.read()

        def __getattr__(self, name):
            return getattr(self.cap, name)

    def fake_open(path, log, hw=True):
        cap = real_open(path, log, hw=False)
        return EarlyStop(cap, 40) if hw else cap

    monkeypatch.setattr(bf, "_open_video", fake_open)
    src = make_video(tmp_path / "clip.mp4", frames=90)
    out = tmp_path / "clip_blurred.mp4"
    logs, report = [], {}
    res = bf.process_video(src, out, StubDetector(), video_args(), log=logs.append, report=report)
    assert res == (90, 0) and report["hw_decode"] is False
    assert any("提前結束" in line for line in logs)


@needs_ffmpeg
def test_ffmpeg_writer_close_times_out_on_hung_process():
    """ffmpeg 卡死時 close() 不會永遠等：超過 timeout 就強制結束並回報。"""
    w = bf._FfmpegWriter.__new__(bf._FfmpegWriter)
    w.encoder = "fake"
    w._err = __import__("tempfile").TemporaryFile()
    w.proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                              stdin=subprocess.PIPE, **bf._subprocess_kwargs())
    t0 = time.time()
    msg = w.close(timeout=2)
    assert msg and "強制中止" in msg and time.time() - t0 < 15
    assert w.proc.poll() is not None


# ---------------------------------------------------------------------------
# 真實模型冒煙（CPU）
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not bf.SCRFD_MODEL.exists() or not bf.HEAD_MODEL.exists(), reason="缺模型檔")
def test_real_models_smoke_cpu():
    import argparse

    args = argparse.Namespace(detector="scrfd", det_size=1280, conf=0.4, head=True, head_conf=0.5, rescue=True,
                              device="cpu", multiscale=True, track=True, track_conf=0.15)
    det = bf.create_detector(args, log=lambda _s: None)
    noise = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    dets = det.detect_scored(noise)  # 走過多尺度、頭部縮小重掃、旋轉補救
    assert isinstance(dets, list) and all(isinstance(d, bf.Detection) for d in dets)
    assert det.detect(np.full((480, 640, 3), 120, np.uint8)) == []
    assert bf.device_label(det) == "CPU"
