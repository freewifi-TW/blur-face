"""GUI 端不需要顯示的邏輯：紀錄輪替、媒體收集、忙碌時的按鈕狀態。需要 PySide6（CI 有裝）。"""
import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import gui  # noqa: E402


def test_rotate_log(tmp_path):
    p = tmp_path / "blurface.log"
    p.write_bytes(b"x" * 1024)
    gui.rotate_log(p, limit=512)
    assert not p.exists() and (tmp_path / "blurface.log.1").exists()
    small = tmp_path / "small.log"
    small.write_bytes(b"x")
    gui.rotate_log(small, limit=512)
    assert small.exists()


def test_collect_media_skips_blurred_and_unknown(tmp_path):
    for name in ("a.jpg", "a_blurred.jpg", "b.MOV", "c.txt", "d.heic"):
        (tmp_path / name).write_bytes(b"")
    got = sorted(p.name for p in gui.collect_media([str(tmp_path)]))
    assert got == ["a.jpg", "b.MOV", "d.heic"]


def test_set_busy_disables_list_editing():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    win = gui.MainWindow()
    win.set_busy(True)
    assert all(not b.isEnabled() for b in win.list_btns) and not win.list.acceptDrops()
    win.set_busy(False)
    assert all(b.isEnabled() for b in win.list_btns) and win.list.acceptDrops()
    win.on_item(99, "不存在的列")  # 不該炸掉
    win.close()
