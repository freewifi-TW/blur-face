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


# --- 更新檢查 ---
def test_parse_version():
    assert gui.parse_version("v1.3.4") == (1, 3, 4)
    assert gui.parse_version("1.10.0") == (1, 10, 0)
    assert gui.parse_version("v2.0.0-beta.1") == (2, 0, 0)
    assert gui.parse_version("dev") is None
    assert gui.parse_version("") is None
    assert gui.parse_version("v1.10.0") > gui.parse_version("v1.9.9")  # 數值比較，不是字串比較


def test_check_update_logic():
    latest = {"tag": "v1.4.0", "url": "https://example.test/v1.4.0"}
    assert gui.check_update("1.3.4", latest) == latest
    assert gui.check_update("1.4.0", latest) is None        # 相同版本
    assert gui.check_update("1.5.0", latest) is None        # 本機更新（開發中）
    assert gui.check_update("dev", latest) is None          # 開發版不提醒
    assert gui.check_update("1.3.4", None) is None          # 抓不到（離線）
    assert gui.check_update("1.3.4", {"tag": "nightly", "url": ""}) is None  # tag 不是版本號
    assert gui.check_update("1.3.4", latest, skipped="v1.4.0") is None   # 使用者略過此版本
    assert gui.check_update("1.3.4", latest, skipped="v1.3.5") == latest  # 略過的是舊版，仍提醒


def test_fetch_latest_release_parses_and_swallows_errors(monkeypatch):
    import io
    import json

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()

    payload = json.dumps({"tag_name": "v9.9.9", "html_url": "https://example.test/rel"}).encode()
    monkeypatch.setattr(gui.urllib.request, "urlopen", lambda req, timeout: Resp(payload))
    assert gui.fetch_latest_release() == {"tag": "v9.9.9", "url": "https://example.test/rel"}

    monkeypatch.setattr(gui.urllib.request, "urlopen", lambda req, timeout: Resp(b"not json"))
    assert gui.fetch_latest_release() is None

    def boom(req, timeout):
        raise OSError("offline")

    monkeypatch.setattr(gui.urllib.request, "urlopen", boom)
    assert gui.fetch_latest_release() is None


def test_update_bar_shows_and_skip_writes_file(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication

    monkeypatch.setattr(gui, "SKIP_FILE", tmp_path / "skip_update.txt")
    monkeypatch.setattr(gui, "LOG_DIR", tmp_path)
    app = QApplication.instance() or QApplication([])
    win = gui.MainWindow(check_updates=False)  # 測試不連網
    assert win.update_checker is None and win.update_bar.isHidden()
    win.on_update_found("v9.9.9", "https://example.test/rel")
    assert not win.update_bar.isHidden()
    assert "v9.9.9" in win.update_lbl.text() and "v9.9.9" in win.log.toPlainText()
    assert win.update_url == "https://example.test/rel"
    win.skip_update()
    assert win.update_bar.isHidden()
    assert gui.read_skipped() == "v9.9.9"
    win.close()


def test_update_check_disabled_for_dev_and_flags(monkeypatch):
    monkeypatch.setattr(gui, "__version__", "dev")
    assert not gui.update_check_enabled()
    monkeypatch.setattr(gui, "__version__", "1.3.4")
    monkeypatch.delenv("BLURFACE_NO_UPDATE_CHECK", raising=False)
    monkeypatch.setattr(gui.sys, "argv", ["gui.py"])
    assert gui.update_check_enabled()
    monkeypatch.setattr(gui.sys, "argv", ["gui.py", "--no-update-check"])
    assert not gui.update_check_enabled()
    monkeypatch.setattr(gui.sys, "argv", ["gui.py"])
    monkeypatch.setenv("BLURFACE_NO_UPDATE_CHECK", "1")
    assert not gui.update_check_enabled()
