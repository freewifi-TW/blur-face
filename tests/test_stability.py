"""打碼區域穩定性：偵測框小幅抖動、頭部框間歇漏偵、頭部移動時，輸出框與馬賽克格子都不該逐幀跳動。"""
import argparse
import random

import numpy as np

import blur_faces as bf

D = bf.Detection


def run(seq, **kw):
    tr = bf.StreamTracker(**kw)
    out = []
    for dets in seq:
        out.extend(b for _, b in tr.push(np.zeros((2, 2, 3), np.uint8), dets))
    out.extend(b for _, b in tr.flush())
    return [fr[0] for fr in out]


def jittery_static(n=90, jitter=12, seed=3):
    """頭幾乎不動：臉框每幀 ±jitter 抖動，頭框每 4 幀漏 1 幀。"""
    random.seed(seed)
    seq = []
    for i in range(n):
        j = lambda: random.randint(-jitter, jitter)  # noqa: E731
        face = (1500 + j(), 900 + j(), 520 + j(), 560 + j())
        head = (1440 + j(), 760 + j(), 640 + j(), 760 + j())
        dets = [D(face, .9, True)] + ([D(head, .8, True)] if i % 4 else [])
        seq.append(bf._merge_scored(dets))
    return seq


def test_static_head_output_is_stable():
    seq = jittery_static()
    raw = [fr[0].box for fr in seq]
    out = run(seq)
    assert len(out) == len(seq) and all(len({b}) for b in out)
    diffs = np.abs(np.diff(np.array(out), axis=0))
    assert diffs.mean() < 3, f"平滑後每幀平均變化 {diffs.mean():.1f}px 仍過大"
    # 逐邊死區：中段（窗完整）只在出現新極值時外擴幾個像素，其餘幀完全不動
    mid = np.array(out[10:-10])
    d = np.abs(np.diff(mid, axis=0))
    assert (d.sum(axis=1) > 0).sum() <= 10, f"70 幀中有 {(d.sum(axis=1) > 0).sum()} 幀在變"
    assert d.max() <= 12 and d.mean() < 0.5, f"單幀最大變化 {d.max()}px、平均 {d.mean():.2f}px"
    # 一定包含當幀的原始偵測框
    for (rx, ry, rw, rh), (ox, oy, ow, oh) in zip(raw, out):
        assert ox <= rx and oy <= ry and ox + ow >= rx + rw and oy + oh >= ry + rh


def test_moving_head_does_not_smear():
    """頭以每幀 25px 移動：輸出框寬度不該比原始框寬多出一大截（單純窗內聯集會拖影）。"""
    random.seed(5)
    seq = []
    for i in range(80):
        j = lambda: random.randint(-8, 8)  # noqa: E731
        seq.append([D((1000 + i * 25 + j(), 800 + j(), 600 + j(), 700 + j()), .9, True)])
    out = run(seq)
    widths = [b[2] for b in out[10:-10]]
    assert max(widths) <= 600 + 16 + 60, f"移動時拖影：最寬 {max(widths)}px"
    # 而且仍每幀都包含原始框
    for fr, (ox, oy, ow, oh) in zip(seq, out):
        rx, ry, rw, rh = fr[0].box
        assert ox <= rx and oy <= ry and ox + ow >= rx + rw and oy + oh >= ry + rh


def test_alternating_face_head_boxes_single_size():
    seq = [[D((500, 400, 200, 200), 0.9, True)] if i % 2 else [D((470, 340, 260, 320), 0.9, True)]
           for i in range(60)]
    out = run(seq)
    assert {(b[2], b[3]) for b in out[8:-8]} == {(260, 320)}


def test_slow_drift_follows_without_smear():
    """每幀 1px 慢速漂移：輸出要跟上（不能永遠凍結），但也不能拖影。"""
    seq = [[D((1000 + i, 800, 600, 700), .9, True)] for i in range(120)]
    out = run(seq)
    assert out[-15][0] - out[15][0] > 60, "死區讓輸出跟不上慢速漂移"
    assert max(b[2] for b in out) <= 600 + 12 + 35  # 前後緣各最多多出 slack（5%）


def test_mosaic_cells_anchored_to_frame_grid():
    """同一畫面上兩個相差幾像素的 ROI，重疊區域內部的馬賽克格子顏色與位置必須完全相同。"""
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    a, b = base.copy(), base.copy()
    bf.censor_region(a, 300, 200, 700, 620, "mosaic", 5, False)
    bf.censor_region(b, 305, 207, 703, 615, "mosaic", 5, False)
    blk = bf.mosaic_block(300, 200, 700, 620, 5)
    assert blk == 32
    inner = (slice(200 + 2 * blk, 615 - 2 * blk), slice(300 + 2 * blk, 700 - 2 * blk))
    assert np.array_equal(a[inner], b[inner])
    # 舊做法（以 ROI 左上角為基準）在同樣位移下幾乎每個像素都不同，這裡順帶確認格子確實有被打碼
    assert not np.array_equal(a[inner], base[inner])


def test_mosaic_block_quantization_and_ellipse():
    assert bf.mosaic_block(0, 0, 600, 600, 5) == 64
    assert bf.mosaic_block(0, 0, 40, 40, 5) == 4
    assert bf.mosaic_block(0, 0, 600, 600, 10) == 128
    frame = np.full((300, 300, 3), 90, np.uint8)
    frame[100:200, 100:200] = 200
    bf.censor_region(frame, 90, 90, 210, 210, "mosaic", 5, True)
    assert frame[91, 91].tolist() == [90, 90, 90]  # 橢圓外角落未動
    assert frame[150, 150, 0] == 200               # 純色內部格子平均仍是 200
    assert 90 < frame[100, 150, 0] < 200           # 跨越邊界的格子被平均


def test_apply_boxes_stays_inside_padded_box():
    args = argparse.Namespace(mode="blur", strength=5, pad=0.0, ellipse=False)
    tex = np.random.default_rng(1).integers(0, 255, (360, 640, 3), dtype=np.uint8)
    ref = tex.copy()
    bf.apply_boxes(tex, [(101, 51, 60, 60)], args)
    changed = np.argwhere((tex != ref).any(axis=2))
    (y0, x0), (y1, x1) = changed.min(axis=0), changed.max(axis=0)
    assert x0 >= 101 and y0 >= 51 and x1 <= 160 and y1 <= 110
