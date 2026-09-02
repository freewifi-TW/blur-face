#!/usr/bin/env python3
"""Blur Face 桌面版 GUI：拖放照片/影片，AI 偵測人臉並打碼。

跑法：.venv/bin/python gui.py
打包後由 PyInstaller 直接啟動本檔。
"""

import argparse
import gc
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSlider, QVBoxLayout, QWidget,
)

from blur_faces import (
    IMAGE_EXTS, VIDEO_EXTS, create_detector, default_output, device_label,
    pick_encoder, process_image, process_video, runtime_info,
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
    sig_status = Signal(str)           # 狀態列文字（實際使用的裝置 / 編碼器）
    sig_log = Signal(str)              # 處理紀錄（一行一則）

    def __init__(self, files: list[Path], args: argparse.Namespace, out_dir: Path | None, detector=None):
        super().__init__()
        self.files = files
        self.args = args
        self.out_dir = out_dir
        self.cancelled = False
        # 設定沒變時沿用上一輪的偵測器：每輪重建 GPU session 會累積未釋放的顯示卡資源，
        # 幾輪後 DirectML 初始化失敗、默默退回 CPU（症狀：GPU 0%、CPU 飆高、速度極慢）
        self.detector = detector

    def cancel(self):
        self.cancelled = True

    def run(self):
        log = self.sig_log.emit
        try:
            if self.detector is None:
                t0 = time.time()
                self.detector = create_detector(self.args, log=log)
                log(f"偵測器就緒（{time.time() - t0:.1f}s），裝置：{device_label(self.detector)}")
            else:
                log(f"沿用上一輪的偵測器，裝置：{device_label(self.detector)}")
        except SystemExit as e:
            log(f"✗ 偵測器建立失敗：{e}")
            self.sig_done.emit(f"錯誤：{e}")
            return
        detector = self.detector
        dev = device_label(detector)
        status = f"偵測 {dev}"
        if self.args.device != "cpu" and dev == "CPU":
            status += "（GPU 不可用，見處理紀錄）"
            log("⚠ 已勾選 GPU 加速但這輪沒有用到 GPU，偵測改在 CPU 執行，速度會慢很多；重開程式通常可恢復")
        if any(p.suffix.lower() in VIDEO_EXTS for p in self.files):
            try:
                encoder = pick_encoder(self.args.encoder)[0]
            except SystemExit as e:
                log(f"✗ {e}")
                self.sig_done.emit(f"錯誤：{e}")
                return
            status += f" · 編碼 {encoder}"
            log(f"影片編碼器：{encoder}")
        self.sig_status.emit(f"處理中…（{status}）")

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
                        log=log, progress=progress,
                        cancel=lambda: self.cancelled,
                    )
                    if result is None:
                        raise RuntimeError("已取消" if self.cancelled else "無法解碼")
                    frames, faces = result
                    self.sig_item.emit(i, f"✓ {faces} 次人臉偵測 / {frames} 幀")
                else:
                    faces = process_image(path, out, detector, self.args, log=log)
                    if faces is None:
                        raise RuntimeError("無法讀取")
                    self.sig_item.emit(i, f"✓ {faces} 張人臉")
                ok += 1
                total_faces += faces
            except Exception as e:
                fail += 1
                log(f"✗ {path.name}：{e}")
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
        self.resize(680, 780)
        self.files: list[Path] = []
        self.worker: Worker | None = None
        self.out_dir: Path | None = None
        self.cached_detector = None   # 跨輪沿用的偵測器（見 Worker.__init__ 說明）；self.detector 是下拉選單
        self.detector_key = None      # 建立該偵測器時的設定，設定變了才重建
        self.logged_env = False

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        # --- 檔案清單 ---
        self.list = DropList()
        self.list.files_dropped.connect(self.add_paths)
        self.list.itemDoubleClicked.connect(self.open_output)
        self.list.setToolTip(
            "支援拖放檔案或整個資料夾（資料夾會自動抓出裡面所有照片和影片）\n"
            "支援格式：jpg/png/webp 等照片，mp4/mov/mkv 等影片"
        )
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
        opts = QGroupBox("設定（滑鼠停在欄位上可看說明）")
        grid = QGridLayout(opts)

        def add_field(label_text: str, widget, row: int, col: int, tip: str, extra=None):
            """加一組「標籤 + 控制項」，兩者都掛上說明 tooltip。"""
            lbl = QLabel(label_text)
            for w in filter(None, (lbl, widget, extra)):
                w.setToolTip(tip)
            grid.addWidget(lbl, row, col)
            grid.addWidget(widget, row, col + 1)
            if extra is not None:
                grid.addWidget(extra, row, col + 2)

        self.mode = QComboBox()
        self.mode.addItems(["馬賽克", "高斯模糊"])
        add_field("打碼方式", self.mode, 0, 0, (
            "馬賽克：把臉變成色塊格子\n"
            "高斯模糊：把臉霧化\n"
            "兩種都能遮蔽身分，馬賽克的遮蔽效果較強"
        ))

        self.strength = QSlider(Qt.Horizontal)
        self.strength.setRange(1, 10)
        self.strength.setValue(5)
        self.strength_lbl = QLabel("5")
        self.strength.valueChanged.connect(lambda v: self.strength_lbl.setText(str(v)))
        add_field("強度", self.strength, 0, 2, (
            "打碼的粗細程度（1–10）\n"
            "越大格子越粗 / 越模糊，越難被辨識\n"
            "機敏內容建議 7 以上"
        ), extra=self.strength_lbl)

        self.detector = QComboBox()
        self.detector.addItems([label for label, _ in DETECTOR_CHOICES])
        add_field("偵測器", self.detector, 1, 0, (
            "高準確度（SCRFD）：側臉、小臉、遮擋臉都抓得到，建議使用\n"
            "快速（YuNet）：速度快約 18 倍，但刁鑽角度容易漏\n"
            "最高召回：兩個模型一起跑取聯集，最慢但最不會漏"
        ))

        self.det_size = QComboBox()
        self.det_size.addItems(["640（快）", "960", "1280（建議）", "1920（小臉多）"])
        self.det_size.setCurrentIndex(2)
        add_field("偵測解析度", self.det_size, 1, 2, (
            "AI 掃描畫面時使用的解析度\n"
            "越高越能抓到畫面中很小的臉，但速度越慢\n"
            "（640 比 1280 快約 4 倍；遠處人很多時選 1920）"
        ))

        self.conf = QSlider(Qt.Horizontal)
        self.conf.setRange(20, 80)
        self.conf.setValue(40)
        self.conf_lbl = QLabel("0.40")
        self.conf.valueChanged.connect(lambda v: self.conf_lbl.setText(f"{v / 100:.2f}"))
        add_field("偵測門檻", self.conf, 2, 0, (
            "AI 認定「這是人臉」所需的最低信心分數（0.20–0.80）\n"
            "調低＝寧可錯殺：模糊臉、小臉也抓，但可能誤打非人臉\n"
            "調高＝只打很確定的臉，但容易漏抓\n"
            "有臉漏抓→調低到 0.25–0.30；風景被亂打碼→調高到 0.50–0.60"
        ), extra=self.conf_lbl)

        self.ellipse = QCheckBox("橢圓遮罩")
        self.ellipse.setToolTip("打碼區域用橢圓形取代矩形，觀感較自然\n（遮蔽範圍比矩形略小）")
        grid.addWidget(self.ellipse, 2, 3)

        self.head = QCheckBox("頭部偵測（含背對鏡頭）")
        self.head.setChecked(True)
        self.head.setToolTip(
            "除了臉之外也偵測「整顆頭」（CrowdHuman 模型）\n"
            "背對鏡頭、側面、低頭的人也會被遮蔽 — 隱私保護建議開啟\n"
            "關閉後只遮有偵測到臉的人"
        )
        grid.addWidget(self.head, 3, 0, 1, 2)

        self.track = QCheckBox("影片追蹤補洞")
        self.track.setChecked(True)
        self.track.setToolTip(
            "影片先整部偵測、把同一個人在時間軸上串成軌跡，\n"
            "短暫漏偵測的幀用前後幀位置自動補齊，避免馬賽克閃爍或漏幀\n"
            "建議開啟；關閉則改為逐幀即時處理"
        )
        grid.addWidget(self.track, 3, 2)

        self.gpu = QCheckBox("GPU 加速偵測")
        self.gpu.setChecked(True)
        self.gpu.setToolTip(
            "用 GPU 跑人臉 / 頭部偵測模型（macOS 走 CoreML、Windows 走 DirectML，內顯也可以）\n"
            "比 CPU 快約 4–6 倍；沒有可用 GPU 時自動改用 CPU，偵測結果完全相同\n"
            "開始處理後狀態列會顯示實際使用的裝置"
        )
        grid.addWidget(self.gpu, 4, 0, 1, 2)

        self.hw_encode = QCheckBox("硬體編碼影片")
        self.hw_encode.setChecked(True)
        self.hw_encode.setToolTip(
            "影片輸出改用顯示卡 / 媒體引擎的 H.264 編碼器（VideoToolbox、NVENC、QSV、AMF）\n"
            "編碼速度快很多，畫質略低於軟體編碼 libx264；沒有可用硬體時自動改用 libx264\n"
            "追求最高畫質可關閉"
        )
        grid.addWidget(self.hw_encode, 4, 2)

        self.out_btn = QPushButton("輸出資料夾：原檔旁（點擊變更）")
        self.out_btn.setToolTip(
            "處理結果的存放位置，原始檔案永遠不會被修改\n"
            "預設：輸出在每個原檔旁邊，檔名加上 _blurred\n"
            "點擊可改成統一輸出到你指定的資料夾"
        )
        self.out_btn.clicked.connect(self.pick_out_dir)
        grid.addWidget(self.out_btn, 5, 0, 1, 5)

        layout.addWidget(opts)

        # --- 進度與控制 ---
        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        # --- 處理紀錄 ---
        log_box = QGroupBox("處理紀錄（每個檔案的耗時、使用的裝置與編碼器；遇到問題可複製回報）")
        log_layout = QVBoxLayout(log_box)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setFixedHeight(150)
        self.log.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.log.setStyleSheet("font-family: Menlo, Consolas, 'Courier New', monospace; font-size: 11px;")
        log_layout.addWidget(self.log)
        log_btns = QHBoxLayout()
        log_btns.addStretch(1)
        copy_btn = QPushButton("複製紀錄")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(self.log.toPlainText()))
        clear_btn = QPushButton("清除紀錄")
        clear_btn.clicked.connect(self.log.clear)
        log_btns.addWidget(copy_btn)
        log_btns.addWidget(clear_btn)
        log_layout.addLayout(log_btns)
        layout.addWidget(log_box)

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

    def open_output(self, item):
        """雙擊已完成的項目 → 用系統預設程式開啟輸出檔。"""
        out = default_output(self.files[self.list.row(item)], self.out_dir)
        if out.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(out)))

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
            head=self.head.isChecked(),
            track=self.track.isChecked(),
            device="auto" if self.gpu.isChecked() else "cpu",
            encoder="auto" if self.hw_encode.isChecked() else "software",
        )

    def append_log(self, text: str):
        self.log.appendPlainText(f"[{datetime.now():%H:%M:%S}] {text}")

    def start(self):
        if not self.files:
            QMessageBox.information(self, "沒有檔案", "請先加入照片或影片")
            return
        self.set_busy(True)
        self.progress.setValue(0)
        self.status.setText("處理中…")
        args = self.build_args()
        if not self.logged_env:
            self.append_log(runtime_info())
            self.logged_env = True
        self.append_log(
            f"開始處理 {len(self.files)} 個檔案 · 偵測器 {args.detector} · 解析度 {args.det_size} · "
            f"門檻 {args.conf:.2f} · 頭部 {'開' if args.head else '關'} · 追蹤 {'開' if args.track else '關'} · "
            f"裝置 {args.device} · 編碼 {args.encoder} · 輸出 {self.out_dir or '原檔旁'}"
        )
        key = (args.detector, args.det_size, args.conf, args.head, args.device)
        if key != self.detector_key:
            if self.cached_detector is not None:
                self.append_log("偵測設定已變更，釋放舊偵測器後重建")
            self.cached_detector = None
            self.detector_key = key
            gc.collect()  # 先把舊 session 的 GPU 資源還回去，再建新的
        self.worker = Worker(list(self.files), args, self.out_dir, self.cached_detector)
        self.worker.sig_item.connect(self.on_item)
        self.worker.sig_overall.connect(self.progress.setValue)
        self.worker.sig_done.connect(self.on_done)
        self.worker.sig_status.connect(self.status.setText)
        self.worker.sig_log.connect(self.append_log)
        self.worker.start()

    def cancel(self):
        if self.worker:
            self.worker.cancel()
            self.status.setText("取消中…")

    def on_item(self, row: int, text: str):
        self.list.item(row).setText(f"{self.files[row].name} — {text}")

    def on_done(self, summary: str):
        if self.worker is not None:
            self.cached_detector = self.worker.detector  # 留給下一輪沿用
        self.append_log(summary)
        self.status.setText(summary)
        self.set_busy(False)

    def closeEvent(self, event):
        """關閉視窗時先取消並等待背景處理結束，避免留下佔著 GPU 記憶體的殭屍程序與 ffmpeg 子程序。"""
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.status.setText("正在停止背景處理…")
            self.worker.wait(30_000)
        self.cached_detector = None
        event.accept()

    def set_busy(self, busy: bool):
        self.start_btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)
        for w in (self.mode, self.strength, self.detector, self.det_size, self.conf,
                  self.ellipse, self.head, self.track, self.gpu, self.hw_encode, self.out_btn):
            w.setEnabled(not busy)


def smoke_test() -> int:
    """打包後的自我檢查：模型載入（含 GPU provider 探測）、偵測、ffmpeg 與編碼器探測都正常才回傳 0。"""
    import numpy as np
    from blur_faces import find_ffmpeg

    app = QApplication(sys.argv)
    win = MainWindow()  # noqa: F841 確認 UI 可建立
    det = create_detector(argparse.Namespace(detector="scrfd", conf=0.4, det_size=640, head=True, device="auto"))
    det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
    assert find_ffmpeg(), "找不到 ffmpeg"
    encoder = pick_encoder("auto")[0]
    if sys.stdout is not None:  # Windows --windowed 模式下 stdout 為 None
        print(f"SMOKE OK  偵測裝置={device_label(det)}  編碼器={encoder}")
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
