#!/usr/bin/env python3
"""Blur Face 桌面版 GUI：拖放照片/影片，AI 偵測人臉並打碼。

跑法：.venv/bin/python gui.py
打包後由 PyInstaller 直接啟動本檔。
"""

import argparse
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QSlider, QVBoxLayout, QWidget,
)

from blur_faces import (
    IMAGE_EXTS, VIDEO_EXTS, create_detector, default_output,
    process_image, process_video,
)

MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS

DETECTOR_CHOICES = [
    ("高準確度（SCRFD，建議）", "scrfd"),
    ("快速（YuNet）", "yunet"),
    ("最高召回（兩者聯集，最慢）", "both"),
]


def collect_media(paths: list[str]) -> list[Path]:
    """把拖進來的檔案/資料夾展開成媒體檔清單。"""
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            out.extend(
                f for f in sorted(path.iterdir())
                if f.suffix.lower() in MEDIA_EXTS and "_blurred" not in f.stem
            )
        elif path.suffix.lower() in MEDIA_EXTS:
            out.append(path)
    return out


class DropList(QListWidget):
    files_dropped = Signal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setSelectionMode(QListWidget.ExtendedSelection)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    dragMoveEvent = dragEnterEvent

    def dropEvent(self, e):
        self.files_dropped.emit([u.toLocalFile() for u in e.mimeData().urls()])


class Worker(QThread):
    sig_item = Signal(int, str)        # (row, 狀態文字)
    sig_overall = Signal(int)          # 整體進度 0-100
    sig_done = Signal(str)             # 完成總結

    def __init__(self, files: list[Path], args: argparse.Namespace, out_dir: Path | None):
        super().__init__()
        self.files = files
        self.args = args
        self.out_dir = out_dir
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def run(self):
        try:
            detector = create_detector(self.args)
        except SystemExit as e:
            self.sig_done.emit(f"錯誤：{e}")
            return

        n_files = len(self.files)
        ok = fail = total_faces = 0
        for i, path in enumerate(self.files):
            if self.cancelled:
                break
            self.sig_item.emit(i, "處理中…")
            if self.out_dir:
                self.out_dir.mkdir(parents=True, exist_ok=True)
            out = default_output(path, self.out_dir)

            def progress(done, total, i=i):
                if total > 0 and done % 5 == 0:
                    pct_file = done / total
                    self.sig_item.emit(i, f"處理中… {pct_file:.0%}")
                    self.sig_overall.emit(int((i + pct_file) / n_files * 100))

            try:
                if path.suffix.lower() in VIDEO_EXTS:
                    result = process_video(
                        path, out, detector, self.args,
                        log=lambda *_: None, progress=progress,
                        cancel=lambda: self.cancelled,
                    )
                    if result is None:
                        raise RuntimeError("已取消" if self.cancelled else "無法解碼")
                    frames, faces = result
                    self.sig_item.emit(i, f"✓ {faces} 次人臉偵測 / {frames} 幀")
                else:
                    faces = process_image(path, out, detector, self.args, log=lambda *_: None)
                    if faces is None:
                        raise RuntimeError("無法讀取")
                    self.sig_item.emit(i, f"✓ {faces} 張人臉")
                ok += 1
                total_faces += faces
            except Exception as e:
                fail += 1
                self.sig_item.emit(i, f"✗ {e}")
            self.sig_overall.emit(int((i + 1) / n_files * 100))

        if self.cancelled:
            self.sig_done.emit(f"已取消（完成 {ok} 個檔案）")
        else:
            msg = f"完成 {ok} 個檔案，共打碼 {total_faces} 張人臉"
            if fail:
                msg += f"；{fail} 個失敗"
            self.sig_done.emit(msg)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Blur Face — AI 人臉打碼")
        self.resize(680, 560)
        self.files: list[Path] = []
        self.worker: Worker | None = None
        self.out_dir: Path | None = None

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        # --- 檔案清單 ---
        self.list = DropList()
        self.list.files_dropped.connect(self.add_paths)
        layout.addWidget(QLabel("把照片、影片或整個資料夾拖到下面："))
        layout.addWidget(self.list, stretch=1)

        btns = QHBoxLayout()
        for text, fn in [("加入檔案", self.pick_files), ("加入資料夾", self.pick_dir),
                         ("移除選取", self.remove_selected), ("清空", self.clear_all)]:
            b = QPushButton(text)
            b.clicked.connect(fn)
            btns.addWidget(b)
        layout.addLayout(btns)

        # --- 參數 ---
        opts = QGroupBox("設定")
        grid = QGridLayout(opts)

        self.mode = QComboBox()
        self.mode.addItems(["馬賽克", "高斯模糊"])
        grid.addWidget(QLabel("打碼方式"), 0, 0)
        grid.addWidget(self.mode, 0, 1)

        self.strength = QSlider(Qt.Horizontal)
        self.strength.setRange(1, 10)
        self.strength.setValue(5)
        self.strength_lbl = QLabel("5")
        self.strength.valueChanged.connect(lambda v: self.strength_lbl.setText(str(v)))
        grid.addWidget(QLabel("強度"), 0, 2)
        grid.addWidget(self.strength, 0, 3)
        grid.addWidget(self.strength_lbl, 0, 4)

        self.detector = QComboBox()
        self.detector.addItems([label for label, _ in DETECTOR_CHOICES])
        grid.addWidget(QLabel("偵測器"), 1, 0)
        grid.addWidget(self.detector, 1, 1)

        self.det_size = QComboBox()
        self.det_size.addItems(["640（快）", "960", "1280（建議）", "1920（小臉多）"])
        self.det_size.setCurrentIndex(2)
        grid.addWidget(QLabel("偵測解析度"), 1, 2)
        grid.addWidget(self.det_size, 1, 3)

        self.conf = QSlider(Qt.Horizontal)
        self.conf.setRange(20, 80)
        self.conf.setValue(40)
        self.conf_lbl = QLabel("0.40")
        self.conf.valueChanged.connect(lambda v: self.conf_lbl.setText(f"{v / 100:.2f}"))
        grid.addWidget(QLabel("偵測門檻"), 2, 0)
        grid.addWidget(self.conf, 2, 1)
        grid.addWidget(self.conf_lbl, 2, 2)

        self.ellipse = QCheckBox("橢圓遮罩")
        grid.addWidget(self.ellipse, 2, 3)

        self.out_btn = QPushButton("輸出資料夾：原檔旁（點擊變更）")
        self.out_btn.clicked.connect(self.pick_out_dir)
        grid.addWidget(self.out_btn, 3, 0, 1, 5)

        layout.addWidget(opts)

        # --- 進度與控制 ---
        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        ctrl = QHBoxLayout()
        self.status = QLabel("就緒")
        ctrl.addWidget(self.status, stretch=1)
        self.start_btn = QPushButton("開始處理")
        self.start_btn.setDefault(True)
        self.start_btn.clicked.connect(self.start)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        ctrl.addWidget(self.start_btn)
        ctrl.addWidget(self.cancel_btn)
        layout.addLayout(ctrl)

    # --- 檔案管理 ---
    def add_paths(self, paths: list[str]):
        for f in collect_media(paths):
            if f not in self.files:
                self.files.append(f)
                self.list.addItem(QListWidgetItem(f"{f.name} — 待處理"))
        self.status.setText(f"共 {len(self.files)} 個檔案")

    def pick_files(self):
        exts = " ".join(f"*{e}" for e in sorted(MEDIA_EXTS))
        paths, _ = QFileDialog.getOpenFileNames(self, "選擇照片或影片", "", f"媒體檔 ({exts})")
        self.add_paths(paths)

    def pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "選擇資料夾")
        if d:
            self.add_paths([d])

    def remove_selected(self):
        for item in self.list.selectedItems():
            row = self.list.row(item)
            self.list.takeItem(row)
            del self.files[row]
        self.status.setText(f"共 {len(self.files)} 個檔案")

    def clear_all(self):
        self.list.clear()
        self.files.clear()
        self.status.setText("就緒")

    def pick_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "選擇輸出資料夾")
        if d:
            self.out_dir = Path(d)
            self.out_btn.setText(f"輸出資料夾：{d}（點擊變更）")

    # --- 執行 ---
    def build_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            mode="mosaic" if self.mode.currentIndex() == 0 else "blur",
            strength=self.strength.value(),
            detector=DETECTOR_CHOICES[self.detector.currentIndex()][1],
            det_size=int(self.det_size.currentText().split("（")[0]),
            conf=self.conf.value() / 100,
            pad=0.15,
            keep=4,
            ellipse=self.ellipse.isChecked(),
            preview=False,
        )

    def start(self):
        if not self.files:
            QMessageBox.information(self, "沒有檔案", "請先加入照片或影片")
            return
        self.set_busy(True)
        self.progress.setValue(0)
        self.status.setText("處理中…")
        self.worker = Worker(list(self.files), self.build_args(), self.out_dir)
        self.worker.sig_item.connect(self.on_item)
        self.worker.sig_overall.connect(self.progress.setValue)
        self.worker.sig_done.connect(self.on_done)
        self.worker.start()

    def cancel(self):
        if self.worker:
            self.worker.cancel()
            self.status.setText("取消中…")

    def on_item(self, row: int, text: str):
        self.list.item(row).setText(f"{self.files[row].name} — {text}")

    def on_done(self, summary: str):
        self.status.setText(summary)
        self.set_busy(False)

    def set_busy(self, busy: bool):
        self.start_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)
        for w in (self.mode, self.strength, self.detector, self.det_size,
                  self.conf, self.ellipse, self.out_btn):
            w.setEnabled(not busy)


def smoke_test() -> int:
    """打包後的自我檢查：模型載入、偵測、ffmpeg 都正常才回傳 0。"""
    import numpy as np
    from blur_faces import ScrfdDetector, find_ffmpeg

    app = QApplication(sys.argv)
    win = MainWindow()  # noqa: F841 確認 UI 可建立
    det = ScrfdDetector(conf=0.4, det_size=640)
    det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
    assert find_ffmpeg(), "找不到 ffmpeg"
    if sys.stdout is not None:  # Windows --windowed 模式下 stdout 為 None
        print("SMOKE OK")
    return 0


def main():
    if "--smoke-test" in sys.argv:
        try:
            sys.exit(smoke_test())
        except Exception:
            traceback.print_exc()
            sys.exit(1)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
