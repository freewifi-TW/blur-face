#!/usr/bin/env python3
"""AI 人臉自動打碼工具：支援照片與影片，偵測到的人臉自動打上馬賽克或高斯模糊。

人臉偵測預設使用 SCRFD-10G（InsightFace）模型，對側臉、小臉、遮擋臉有高召回率；
另提供輕量的 YuNet 與兩者聯集（both）模式。全程離線處理，檔案不會上傳。
偵測預設自動使用 GPU（macOS CoreML / Windows DirectML），影片編碼可用硬體編碼器；
影片處理完成後若有 ffmpeg，會直接把原始音軌接回輸出檔。
"""

import argparse
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# 在虛擬桌面 / 串流環境（如 Virtual Desktop）中，OpenCV 的 OpenCL(T-API) 可能被導向虛擬顯示卡，
# 使 cv2.resize 等運算每次呼叫都極慢。我們一律傳 numpy Mat、本來就走 CPU，關掉 OpenCL 無副作用。
try:
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(False)
except Exception:
    pass


def process_rss_mb() -> float | None:
    """回傳目前行程的常駐記憶體（MB），best-effort、跨平台、不需 psutil；取不到回傳 None。"""
    if sys.platform == "win32":
        try:
            import ctypes
            class _PMC(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            c = _PMC(); c.cb = ctypes.sizeof(_PMC)
            h = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
                return c.WorkingSetSize / (1024 * 1024)
        except Exception:
            return None
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # mac: bytes；linux: KB（此為峰值）
        return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024
    except Exception:
        return None


def resource_path(rel: str) -> Path:
    """開發時相對於原始碼；PyInstaller 打包後相對於解包目錄。"""
    return Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / rel


def find_ffmpeg() -> str | None:
    """優先用系統 ffmpeg，否則用 imageio-ffmpeg 內建的靜態版本。"""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _subprocess_kwargs() -> dict:
    """Windows 的 GUI 程式啟動 console 子程序會閃出主控台視窗，加旗標隱藏。"""
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}


MODELS_DIR = resource_path("models")
SCRFD_MODEL = MODELS_DIR / "det_10g.onnx"
YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
HEAD_MODEL = MODELS_DIR / "crowdhuman_yolov5m.onnx"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

DEVICE_CHOICES = ("auto", "cpu", "gpu")
ENCODER_CHOICES = ("auto", "software", "hardware")


# ---------------------------------------------------------------------------
# 推論裝置（onnxruntime execution provider）
# ---------------------------------------------------------------------------

# GPU provider 的嘗試順序與選項；只會用到本機 onnxruntime 有編進去的那些。
_GPU_PROVIDERS = [
    # macOS：CoreML。MLProgram + CPUAndGPU 在 M 系列實測最快、載入最短，且輸出與 CPU 完全一致；
    # ALL 會把部分層丟給 Neural Engine，這兩個模型在 ANE 上反而更慢。
    ("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "CPUAndGPU"}),
    # Windows：DirectML（onnxruntime-directml），NVIDIA / AMD / Intel 含內顯都能用，不需另裝 CUDA。
    ("DmlExecutionProvider", {}),
    # 有另外裝 onnxruntime-gpu 與 CUDA 的環境。
    ("CUDAExecutionProvider", {}),
]

_DEVICE_LABELS = {
    "CoreMLExecutionProvider": "GPU（CoreML）",
    "DmlExecutionProvider": "GPU（DirectML）",
    "CUDAExecutionProvider": "GPU（CUDA）",
    "CPUExecutionProvider": "CPU",
}


def _static_shape_model(model_path: Path, shape: tuple[int, ...]) -> bytes | None:
    """把 ONNX 模型的動態輸入尺寸固定成 shape。

    CoreML / DirectML 需要靜態尺寸才能把整張圖交給 GPU（SCRFD 的輸入是 [1,3,?,?]，
    直接丟給 CoreML 會在 reshape 節點失敗）。輸入本來就是靜態時回傳 None，直接用原檔即可。
    """
    import onnx

    model = onnx.load(str(model_path))
    dims = model.graph.input[0].type.tensor_type.shape.dim
    if all(d.HasField("dim_value") for d in dims):
        return None
    for d, v in zip(dims, shape):
        d.ClearField("dim_param")
        d.dim_value = v
    del model.graph.value_info[:]  # 清掉舊的中間層形狀，讓 shape inference 依新輸入重推
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass
    return model.SerializeToString()


def create_session(model_path: Path, device: str, input_shape: tuple[int, ...], log=None):
    """建立 onnxruntime 推論 session，回傳 (session, provider 名稱)。

    device="auto" / "gpu" 時依序嘗試本機可用的 GPU provider；每個都先用零輸入實際推論一次
    （CoreML 的不相容要到第一次推論才會爆）。全部失敗時 auto 退回 CPU，gpu 直接報錯。
    log(str) 會收到每個 provider 的嘗試結果，方便事後判斷為什麼沒用到 GPU。
    """
    import onnxruntime as ort

    log = log or (lambda *_: None)
    label = f"{model_path.name}@{input_shape[-1]}"

    def make(providers, model=None):
        so = ort.SessionOptions()
        so.log_severity_level = 3  # 關掉無害的 shape / 節點回退警告
        if any(isinstance(p, tuple) and p[0] == "DmlExecutionProvider" for p in providers):
            # DirectML 不支援 memory pattern 與平行執行，官方要求關閉
            so.enable_mem_pattern = False
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        return ort.InferenceSession(
            model if model is not None else str(model_path), sess_options=so, providers=providers
        )

    if device != "cpu":
        available = set(ort.get_available_providers())
        errors = []
        tried = False
        for name, opts in _GPU_PROVIDERS:
            if name not in available:
                continue
            tried = True
            try:
                t0 = time.time()
                model = _static_shape_model(model_path, input_shape)
                sess = make([(name, opts), "CPUExecutionProvider"], model)
                inp = sess.get_inputs()[0]
                sess.run(None, {inp.name: np.zeros(input_shape, dtype=np.float32)})
                log(f"{label}：使用 {_DEVICE_LABELS.get(name, name)}（初始化 {time.time() - t0:.1f}s）")
                return sess, name
            except Exception as e:  # noqa: BLE001 — 任何失敗都改試下一個 provider
                msg = f"{name}: {str(e).splitlines()[0][:160]}"
                errors.append(msg)
                log(f"⚠ {label}：GPU provider 失敗，{msg}")
        if not tried:
            log(f"⚠ 這個 onnxruntime 沒有 GPU provider（可用：{', '.join(sorted(available))}）")
        if device == "gpu":
            detail = "；".join(errors) if errors else (
                "onnxruntime 沒有可用的 GPU provider（Windows 請安裝 onnxruntime-directml）"
            )
            sys.exit(f"GPU 初始化失敗：{detail}")
        log(f"{label}：改用 CPU")
    return make(["CPUExecutionProvider"]), "CPUExecutionProvider"


def runtime_info() -> str:
    """一行執行環境摘要（給 log 用）：onnxruntime 版本與 provider、OpenCV 版本、ffmpeg 路徑。"""
    import platform

    try:
        import onnxruntime as ort

        ort_info = f"onnxruntime {ort.__version__}（{', '.join(ort.get_available_providers())}）"
    except Exception as e:  # noqa: BLE001
        ort_info = f"onnxruntime 無法載入：{e}"
    gpu_info = ""
    if sys.platform == "win32":
        # 列出顯示卡型號與驅動版本，判斷 DirectML 用的是哪張卡（雙顯卡筆電）與 VRAM 是否夠
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + ' (驅動 ' + $_.DriverVersion + ')' }"],
                capture_output=True, text=True, timeout=10, **_subprocess_kwargs(),
            ).stdout
            gpus = [l.strip() for l in out.splitlines() if l.strip()]
            gpu_info = f" · 顯示卡 {' / '.join(gpus) or '未知'}"
        except Exception:  # noqa: BLE001
            gpu_info = " · 顯示卡 查詢失敗"
    try:
        ocl = "關" if cv2.ocl.haveOpenCL() else "無"  # 我們主動關閉，顯示「關」代表偵測得到但已停用
    except Exception:
        ocl = "?"
    rss = process_rss_mb()
    mem = f" · 記憶體 {rss:.0f}MB" if rss else ""
    return (f"{platform.system()} {platform.release()} · Python {platform.python_version()} · "
            f"{ort_info} · OpenCV {cv2.__version__}（OpenCL {ocl}）· ffmpeg {find_ffmpeg() or '無'}{gpu_info}{mem}")


def device_label(detector) -> str:
    """回傳偵測器實際使用的運算裝置描述，例如「GPU（CoreML）」或「CPU」。"""
    if isinstance(detector, RescueDetector):
        return device_label(detector.primary)
    if isinstance(detector, UnionDetector):
        labels = {device_label(d) for d in detector.detectors}
        gpu = sorted(l for l in labels if l != "CPU")
        return gpu[0] if gpu else "CPU"
    return _DEVICE_LABELS.get(getattr(detector, "provider", "CPUExecutionProvider"), "CPU")


# ---------------------------------------------------------------------------
# 偵測器
# ---------------------------------------------------------------------------

class ScrfdDetector:
    """SCRFD-10G 偵測器（onnxruntime），高召回率，對側臉/小臉/遮擋臉表現好。"""

    STRIDES = (8, 16, 32)
    NUM_ANCHORS = 2
    NMS_IOU = 0.4

    def __init__(self, conf: float, det_size: int, device: str = "auto", log=None):
        if not SCRFD_MODEL.exists():
            sys.exit(f"找不到模型檔 {SCRFD_MODEL}，請先下載（見 README.md）")
        self.conf = conf
        self.det_size = max(64, (det_size + 31) // 32 * 32)  # 需為 32 的倍數
        self.session, self.provider = create_session(
            SCRFD_MODEL, device, (1, 3, self.det_size, self.det_size), log
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self._anchor_cache: dict[int, np.ndarray] = {}

    def _anchors(self, stride: int) -> np.ndarray:
        if stride not in self._anchor_cache:
            g = self.det_size // stride
            centers = np.stack(np.mgrid[:g, :g][::-1], axis=-1).astype(np.float32) * stride
            # 每個位置 2 個 anchor，與模型輸出的列順序一致
            self._anchor_cache[stride] = np.repeat(centers.reshape(-1, 2), self.NUM_ANCHORS, axis=0)
        return self._anchor_cache[stride]

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        # 等比例縮放到 det_size x det_size 畫布（右/下留黑邊）
        scale = min(self.det_size / h, self.det_size / w)
        nh, nw = int(h * scale), int(w * scale)
        canvas = np.zeros((self.det_size, self.det_size, 3), dtype=np.uint8)
        canvas[:nh, :nw] = cv2.resize(frame, (nw, nh))

        blob = cv2.dnn.blobFromImage(
            canvas, 1.0 / 128, (self.det_size, self.det_size), (127.5, 127.5, 127.5), swapRB=True
        )
        outs = self.session.run(self.output_names, {self.input_name: blob})

        all_boxes, all_scores = [], []
        for idx, stride in enumerate(self.STRIDES):
            scores = outs[idx].flatten()
            keep = scores >= self.conf
            if not keep.any():
                continue
            centers = self._anchors(stride)[keep]
            dist = outs[idx + 3][keep] * stride  # 到框四邊的距離
            x1 = centers[:, 0] - dist[:, 0]
            y1 = centers[:, 1] - dist[:, 1]
            x2 = centers[:, 0] + dist[:, 2]
            y2 = centers[:, 1] + dist[:, 3]
            boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1) / scale
            all_boxes.extend(boxes.tolist())
            all_scores.extend(scores[keep].tolist())

        if not all_boxes:
            return []
        idxs = cv2.dnn.NMSBoxes(all_boxes, all_scores, self.conf, self.NMS_IOU)
        return [tuple(int(v) for v in all_boxes[i]) for i in np.array(idxs).flatten()]


class YunetDetector:
    """YuNet 偵測器（OpenCV 內建 DNN，只能跑 CPU），輕量快速。"""

    MAX_SIDE = 1280  # 偵測時長邊縮到此尺寸以內
    provider = "CPUExecutionProvider"

    def __init__(self, conf: float):
        if not YUNET_MODEL.exists():
            sys.exit(f"找不到模型檔 {YUNET_MODEL}，請先下載（見 README.md）")
        self.detector = cv2.FaceDetectorYN.create(
            str(YUNET_MODEL), "", (320, 320), score_threshold=conf, nms_threshold=0.3, top_k=5000
        )

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        scale = 1.0
        det_input = frame
        if max(h, w) > self.MAX_SIDE:
            scale = self.MAX_SIDE / max(h, w)
            det_input = cv2.resize(frame, (int(w * scale), int(h * scale)))

        self.detector.setInputSize((det_input.shape[1], det_input.shape[0]))
        _, faces = self.detector.detect(det_input)
        if faces is None:
            return []
        return [tuple((face[:4] / scale).astype(int)) for face in faces]


class HeadDetector:
    """YOLOv5m（CrowdHuman 頭部類別）偵測器：抓「整顆頭」，背對鏡頭、極端角度也偵測得到。"""

    INPUT = 640
    NMS_IOU = 0.45

    def __init__(self, conf: float, device: str = "auto", log=None):
        if not HEAD_MODEL.exists():
            sys.exit(f"找不到模型檔 {HEAD_MODEL}，請先下載（見 README.md）")
        self.session, self.provider = create_session(HEAD_MODEL, device, (1, 3, self.INPUT, self.INPUT), log)
        self.input_name = self.session.get_inputs()[0].name
        self.conf = conf

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        scale = min(self.INPUT / h, self.INPUT / w)
        nh, nw = int(h * scale), int(w * scale)
        canvas = np.full((self.INPUT, self.INPUT, 3), 114, dtype=np.uint8)
        canvas[:nh, :nw] = cv2.resize(frame, (nw, nh))
        blob = np.ascontiguousarray(
            canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32
        ) / 255.0

        # 輸出 (25200, 7)：cx, cy, w, h, objectness, person 分數, head 分數
        out = self.session.run(None, {self.input_name: blob})[0][0]
        scores = out[:, 4] * out[:, 6]  # 只取 head 類別
        keep = scores >= self.conf
        if not keep.any():
            return []
        cx, cy, bw, bh = out[keep, 0], out[keep, 1], out[keep, 2], out[keep, 3]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], axis=1) / scale
        idxs = cv2.dnn.NMSBoxes(boxes.tolist(), scores[keep].tolist(), self.conf, self.NMS_IOU)
        return [tuple(int(v) for v in boxes[i]) for i in np.array(idxs).flatten()]


class UnionDetector:
    """聯集多個偵測器的結果（重疊框合併），追求最高召回率。"""

    def __init__(self, detectors: list):
        self.detectors = detectors
        # 各子偵測器需要的輸入長邊上限。detect() 先把原始幀縮到這個尺寸「一次」讓大家共用，
        # 4K 幀不再被每個子偵測器各自縮圖一遍（4K→1280 一次約 5-10ms，三個模型省兩次）
        self.max_side = max(
            getattr(d, "det_size", None) or getattr(d, "INPUT", None) or getattr(d, "MAX_SIDE", 1280)
            for d in detectors
        )

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = frame.shape[:2]
        s = min(1.0, self.max_side / max(h, w))
        src = frame
        if s < 1.0:
            src = cv2.resize(frame, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        boxes = [b for d in self.detectors for b in d.detect(src)]
        if s < 1.0:
            boxes = [tuple(int(v / s) for v in b) for b in boxes]
        if not boxes:
            return []
        # 用 NMS 去除高度重疊的重複框，分數一律 1.0（只做去重）
        idxs = cv2.dnn.NMSBoxes([list(map(float, b)) for b in boxes], [1.0] * len(boxes), 0.0, 0.5)
        return [boxes[i] for i in np.array(idxs).flatten()]


class RescueDetector:
    """旋轉補救：主偵測器整幀沒抓到臉時，把畫面轉 90 / 270 度再用臉部模型重跑。

    專救橫躺、大角度歪斜的臉——SCRFD 對平面內旋轉約 ±30 度內穩定，躺姿、畫面橫著拍就會漏。
    旋轉重跑只用臉部模型（rescue_parts，預設同主偵測器）：頭部模型對旋轉畫面上的圓弧皮膚、
    關節、物體易給出 0.3-0.6 的誤判（實測誤打碼全部來自它），且頭部偵測本就較耐旋轉，
    正著沒抓到的頭轉了也救不回多少，排除它可把補救誤框壓到 SCRFD 本身的水準。
    命中過的旋轉方向會排到最前、一中就停：連續橫躺片段每幀只多一次偵測，而非固定兩次。
    """

    ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}

    def __init__(self, primary, rescue_parts: list | None = None):
        self.primary = primary
        parts = rescue_parts or [primary]
        self.rescue_det = parts[0] if len(parts) == 1 else UnionDetector(parts)
        self.order = list(self.ROTATIONS)  # 嘗試順序，最近命中的排前面
        self.triggered = 0        # 主偵測沒抓到、啟用補救的幀數
        self.rescued_frames = 0   # 補救有抓到的幀數
        self.rescued_boxes = 0    # 補救抓到的框數

    @staticmethod
    def _unrotate(box, rot: int, h0: int, w0: int) -> tuple[int, int, int, int]:
        """把旋轉後影像座標的框 (x, y, w, h) 映回原圖；h0, w0 為原圖尺寸。"""
        x, y, bw, bh = box
        corners = [(x, y), (x + bw, y), (x, y + bh), (x + bw, y + bh)]
        if rot == 90:      # 原 (x, y) → 順時針轉後 (h0 - y, x)
            corners = [(py, h0 - px) for px, py in corners]
        elif rot == 270:   # 原 (x, y) → 逆時針轉後 (y, w0 - x)
            corners = [(w0 - py, px) for px, py in corners]
        xs, ys = [c[0] for c in corners], [c[1] for c in corners]
        return int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys))

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        boxes = self.primary.detect(frame)
        if boxes:
            return boxes
        self.triggered += 1
        h0, w0 = frame.shape[:2]
        for rot in list(self.order):
            found = self.rescue_det.detect(cv2.rotate(frame, self.ROTATIONS[rot]))
            if found:
                self.order.remove(rot)
                self.order.insert(0, rot)
                self.rescued_frames += 1
                self.rescued_boxes += len(found)
                return [self._unrotate(b, rot, h0, w0) for b in found]
        return []

    def stats(self) -> tuple[int, int, int]:
        return self.triggered, self.rescued_frames, self.rescued_boxes


def rescue_stats(detector) -> tuple[int, int, int] | None:
    """回傳補救統計 (觸發幀數, 救回幀數, 救回框數)；偵測器沒開補救時回傳 None。"""
    return detector.stats() if isinstance(detector, RescueDetector) else None


def create_detector(args, log=None):
    """依 args 建立偵測器；log(str) 會收到各模型實際使用的裝置與 GPU 初始化失敗原因。

    - detector=scrfd / both 時預設多尺度：det-size 高於 640 會再加一道 640 掃描取聯集。SCRFD 的
      訓練畫布是 640、看過的臉最大約 500px，特寫大臉在高解析度掃描下會超出尺度而分數崩掉，
      640 那道把它補回來（實測 9 張側臉照從 5 張提升到 8 張），GPU 每幀只多約 14 ms。args.multiscale=False 停用。
    - args.head_conf（預設 0.5）：頭部模型的門檻，與人臉門檻獨立。圓弧物體（椅端、門把、燈罩）多在
      0.5 到 0.6 之間被誤判成頭，室內誤框多時可調到 0.6；真人的頭在中等尺寸下多為 0.65 以上。
    - args.rescue（預設關）：整幀沒抓到臉時，把畫面轉 90/270 度用「臉部模型」再跑一次（旋轉補救），
      不含頭部模型——頭部模型對旋轉畫面的圓弧皮膚/物體易誤判，且頭部偵測本就耐旋轉。
      救回橫躺、大角度歪斜的臉，誤框率為 SCRFD 本身的水準。
    """
    device = getattr(args, "device", "auto")
    parts = []
    if args.detector in ("scrfd", "both"):
        parts.append(ScrfdDetector(args.conf, args.det_size, device, log))
        if getattr(args, "multiscale", True) and parts[-1].det_size > 640:
            parts.append(ScrfdDetector(args.conf, 640, device, log))
    if args.detector in ("yunet", "both"):
        parts.append(YunetDetector(args.conf))
        if log and device != "cpu" and args.detector == "yunet":
            log("YuNet 走 OpenCV DNN，只能用 CPU")
    face_parts = list(parts)  # 旋轉補救只重跑臉部模型（不含頭部），共用同一批 session
    if getattr(args, "head", False):
        parts.append(HeadDetector(getattr(args, "head_conf", 0.5), device, log))
    det = parts[0] if len(parts) == 1 else UnionDetector(parts)
    if getattr(args, "rescue", False):
        det = RescueDetector(det, face_parts)
    return det


# ---------------------------------------------------------------------------
# 打碼
# ---------------------------------------------------------------------------

def expand_box(box, pad: float, frame_shape) -> tuple[int, int, int, int]:
    """把偵測框往外擴 pad 比例，並裁在畫面範圍內。"""
    h, w = frame_shape[:2]
    x, y, bw, bh = box
    dx, dy = int(bw * pad), int(bh * pad)
    x1 = max(0, x - dx)
    y1 = max(0, y - dy)
    x2 = min(w, x + bw + dx)
    y2 = min(h, y + bh + dy)
    return x1, y1, x2, y2


def censor_region(frame, x1, y1, x2, y2, mode: str, strength: int, ellipse: bool):
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return

    if mode == "mosaic":
        # strength 越大格子越大（越模糊）；至少保留 3 格
        blocks = max(3, 18 - strength * 2)
        small = cv2.resize(roi, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
        censored = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:  # blur
        # kernel 跟著 ROI 尺寸走才有足夠遮蔽力，但大 ROI 直接模糊極慢（4K 滿版臉 k≈550，
        # 單框要 1.2s）；先把長邊縮到 256 再用等比例 kernel 模糊、放大回去，
        # 相對模糊程度相同（k/邊長 不變）、視覺上一樣無法辨識，只要約 3ms。
        src = roi
        scale = 256 / max(roi.shape[:2])
        if scale < 1.0:
            src = cv2.resize(
                roi, (max(1, int(roi.shape[1] * scale)), max(1, int(roi.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        k = max(5, (max(src.shape[:2]) // (12 - strength)) | 1)  # 奇數 kernel
        censored = cv2.GaussianBlur(src, (k, k), 0)
        if src is not roi:
            censored = cv2.resize(censored, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)

    if ellipse:
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.ellipse(
            mask,
            (roi.shape[1] // 2, roi.shape[0] // 2),
            (roi.shape[1] // 2, roi.shape[0] // 2),
            0, 0, 360, 255, -1,
        )
        roi[mask > 0] = censored[mask > 0]
    else:
        frame[y1:y2, x1:x2] = censored


def apply_boxes(frame, boxes, args):
    """把一組偵測框打碼到畫面上。"""
    for box in boxes:
        x1, y1, x2, y2 = expand_box(box, args.pad, frame.shape)
        censor_region(frame, x1, y1, x2, y2, args.mode, args.strength, args.ellipse)


def process_frame(frame, detector, args, sticky_boxes: list | None = None):
    """偵測並打碼單一畫面。sticky_boxes 用於影片的跨幀補償，避免偶爾漏偵測造成閃爍。

    回傳本幀偵測到的人臉數。
    """
    boxes = detector.detect(frame)

    if sticky_boxes is not None:
        # 本幀偵測到的框保留 args.keep 幀，短暫漏偵測時仍持續遮蔽
        sticky_boxes[:] = [(b, ttl - 1) for b, ttl in sticky_boxes if ttl > 1]
        sticky_boxes.extend((b, args.keep) for b in boxes)
        draw_boxes = [b for b, _ in sticky_boxes]
    else:
        draw_boxes = boxes

    apply_boxes(frame, draw_boxes, args)
    return len(boxes)


# ---------------------------------------------------------------------------
# 影片追蹤補洞（線上版）
# ---------------------------------------------------------------------------

def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class StreamTracker:
    """線上版「追蹤補洞」：逐幀送進畫面與偵測框，延遲 delay 幀後吐出遮蔽框已確定的畫面。

    同一個人跨幀以 IoU 串成軌跡；軌跡中短暫漏偵測（≤ max_gap 幀）的幀用前後幀線性內插補齊，
    軌跡起點往前、終點往後各延伸 extend 幀。結果與「全片偵測完再回頭補洞」的離線做法完全相同，
    但只需暫存 delay 幀畫面，影片只要解碼一遍。

    min_hits：一條軌跡至少要被偵測到幾幀才輸出（預設 2）。只出現一幀的框幾乎都是誤判（圓弧物體、
    紋理），而追蹤補洞會把它往前後各延伸 extend 幀、放大成 13 幀的馬賽克塊；真臉幾乎每幀都會被
    偵測到，不受影響。
    """

    def __init__(self, iou_thresh: float = 0.3, max_gap: int = 15, extend: int = 6, min_hits: int = 2):
        self.iou_thresh = iou_thresh
        self.max_gap = max_gap
        self.extend = extend
        self.min_hits = max(1, min_hits)
        # 內插最多回頭改 max_gap-1 幀、起點延伸最多回頭 extend 幀；一條軌跡要等 max_gap 幀沒續接
        # 才能確定結束（終點延伸），所以延遲 max_gap+1 幀後該幀的框就全部確定了。
        # 有最少命中數要求時，起點往前延伸到幀 e 的軌跡最晚在 e+extend 才首次出現、再 max_gap 幀
        # 才知道有沒有第二次偵測，所以要多等 extend 幀。
        self.delay = max_gap + 1 + (extend if self.min_hits > 1 else 0)
        self.tracks: list[dict] = []        # {"boxes": {幀號: box}, "last": 最後偵測到的幀號, "hits": 偵測次數}
        self.pending: deque = deque()       # 尚未輸出的 (幀號, 畫面)
        self.boxes: dict[int, list] = {}    # 幀號 -> [(軌跡, 框)]（偵測 + 內插 + 起點延伸），輸出時依 hits 過濾
        self.idx = 0

    def push(self, frame: np.ndarray, detections: list) -> list[tuple[np.ndarray, list]]:
        """送入一幀與其偵測框，回傳此時已確定、可輸出的 [(畫面, 遮蔽框), ...]。"""
        idx = self.idx
        self.idx += 1
        self.pending.append((idx, frame))
        frame_boxes = self.boxes.setdefault(idx, [])

        active = [t for t in self.tracks if idx - t["last"] <= self.max_gap]
        pairs = sorted(
            ((_iou(t["boxes"][t["last"]], b), ti, bi)
             for ti, t in enumerate(active) for bi, b in enumerate(detections)),
            reverse=True,
        )
        used_t, used_b = set(), set()
        for iou_v, ti, bi in pairs:
            if iou_v < self.iou_thresh:
                break
            if ti in used_t or bi in used_b:
                continue
            t, box = active[ti], detections[bi]
            a, box_a = t["last"], t["boxes"][t["last"]]
            for i in range(a + 1, idx):  # 漏偵測的幀：線性內插
                w = (i - a) / (idx - a)
                self.boxes[i].append((t, tuple(
                    int(round(pa * (1 - w) + pb * w)) for pa, pb in zip(box_a, box)
                )))
            t["boxes"][idx] = box
            t["last"] = idx
            t["hits"] += 1
            frame_boxes.append((t, box))
            used_t.add(ti)
            used_b.add(bi)
        for bi, box in enumerate(detections):
            if bi in used_b:
                continue
            t = {"boxes": {idx: box}, "last": idx, "hits": 1}
            self.tracks.append(t)
            frame_boxes.append((t, box))
            for i in range(max(0, idx - self.extend), idx):  # 軌跡起點往前延伸
                self.boxes[i].append((t, box))

        out = self._emit(idx - self.delay)
        # 已結束且終點延伸也輸出完的軌跡可以丟掉
        self.tracks = [t for t in self.tracks if idx - t["last"] <= self.delay + self.extend]
        return out

    def flush(self) -> list[tuple[np.ndarray, list]]:
        """影片結束：輸出所有還在緩衝的畫面。"""
        return self._emit(self.idx - 1)

    def _emit(self, upto: int) -> list[tuple[np.ndarray, list]]:
        out = []
        while self.pending and self.pending[0][0] <= upto:
            e, frame = self.pending.popleft()
            boxes = [b for t, b in self.boxes.pop(e, []) if t["hits"] >= self.min_hits]
            for t in self.tracks:  # 已結束軌跡的終點往後延伸
                a = t["last"]
                if a < e <= a + self.extend and t["hits"] >= self.min_hits:
                    boxes.append(t["boxes"][a])
            out.append((frame, boxes))
        return out


# ---------------------------------------------------------------------------
# 照片
# ---------------------------------------------------------------------------

def process_image(path: Path, out_path: Path, detector, args, log=print):
    img = cv2.imread(str(path))
    if img is None:
        log(f"⚠ 無法讀取圖片：{path}")
        return None
    t0 = time.time()
    before = rescue_stats(detector)
    n = process_frame(img, detector, args)
    cv2.imwrite(str(out_path), img)
    after = rescue_stats(detector)
    note = "，旋轉補救" if before and after and after[1] > before[1] else ""
    log(f"✓ {path.name} → {out_path.name}（{img.shape[1]}x{img.shape[0]}，偵測到 {n} 張人臉{note}，{time.time() - t0:.2f}s）")
    return n


# ---------------------------------------------------------------------------
# 影片編碼
# ---------------------------------------------------------------------------

_SOFTWARE_ENCODER = ("libx264", ["-preset", "medium", "-crf", "20"])

# 各平台硬體編碼器（依優先順序）與畫質參數，畫質設定約略對齊 libx264 crf 20
_HW_ENCODERS = {
    "darwin": [("h264_videotoolbox", ["-q:v", "65"])],
    "win32": [
        ("h264_nvenc", ["-preset", "p4", "-rc", "vbr", "-cq", "23", "-b:v", "0"]),
        ("h264_qsv", ["-global_quality", "23"]),
        ("h264_amf", ["-rc", "cqp", "-qp_i", "22", "-qp_p", "24"]),
    ],
    "linux": [
        ("h264_nvenc", ["-preset", "p4", "-rc", "vbr", "-cq", "23", "-b:v", "0"]),
        ("h264_qsv", ["-global_quality", "23"]),
    ],
}
_encoder_cache: dict[str, tuple[str, list[str]]] = {}


def _probe_encoder(ffmpeg: str, name: str, opts: list[str]) -> bool:
    """實際試編一小段黑畫面：ffmpeg 有編進某個硬體編碼器，不代表這台機器有對應硬體。"""
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=black:s=256x256:r=30:d=0.2",
        "-c:v", name, *opts, "-pix_fmt", "yuv420p", "-f", "null", "-",
    ]
    try:
        return subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30, **_subprocess_kwargs()
        ).returncode == 0
    except Exception:
        return False


def pick_encoder(mode: str = "auto") -> tuple[str, list[str]]:
    """選擇影片編碼器，回傳 (編碼器名稱, 額外參數)。

    auto：有可用的硬體編碼器就用，否則 libx264；software：一律 libx264；
    hardware：沒有可用的硬體編碼器就報錯。探測結果會快取，每個程序只試一次。
    """
    if mode == "software":
        return _SOFTWARE_ENCODER
    if mode not in _encoder_cache:
        ffmpeg = find_ffmpeg()
        found = None
        if ffmpeg:
            for name, opts in _HW_ENCODERS.get(sys.platform, []):
                if _probe_encoder(ffmpeg, name, opts):
                    found = (name, opts)
                    break
        if found is None and mode == "hardware":
            sys.exit("找不到可用的硬體編碼器（需要 ffmpeg 與支援的 GPU / 媒體引擎），可改用 --encoder auto")
        _encoder_cache[mode] = found or _SOFTWARE_ENCODER
    return _encoder_cache[mode]


class _FfmpegWriter:
    """把 BGR 畫面經 stdin 送給 ffmpeg，一次完成編碼與原始音軌合併（不經中間檔、不重複壓縮）。"""

    def __init__(self, ffmpeg: str, out_path: Path, original: Path, fps: float,
                 size: tuple[int, int], encoder: tuple[str, list[str]]):
        w, h = size
        self.encoder, opts = encoder
        self._err = tempfile.TemporaryFile()
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-framerate", f"{fps:.6f}",
            "-i", "pipe:0",
            "-i", str(original),
            "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", self.encoder, *opts,
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # H.264 yuv420p 需要偶數邊長
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", "-movflags", "+faststart",
            str(out_path),
        ]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._err, **_subprocess_kwargs()
        )

    def write(self, frame: np.ndarray):
        self.write_raw(frame.tobytes())

    def write_raw(self, raw: bytes):
        self.proc.stdin.write(raw)  # ffmpeg 掛掉時會丟 BrokenPipeError

    def close(self, abort: bool = False) -> str | None:
        """結束編碼並回傳 ffmpeg 的錯誤訊息（成功為 None）。abort=True 時直接中止。"""
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        if abort:
            self.proc.terminate()
        rc = self.proc.wait()
        self._err.seek(0)
        msg = self._err.read().decode(errors="replace").strip()
        self._err.close()
        return None if rc == 0 else (msg or f"ffmpeg 結束碼 {rc}")


class _Cv2Writer:
    """沒有 ffmpeg 時的備援：OpenCV 直接輸出 mp4v，無音軌。"""

    encoder = "mp4v（OpenCV，無音軌）"

    def __init__(self, out_path: Path, fps: float, size: tuple[int, int]):
        self.writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, frame: np.ndarray):
        self.writer.write(frame)

    def close(self, abort: bool = False) -> str | None:
        self.writer.release()
        return None


# ---------------------------------------------------------------------------
# 影片處理
# ---------------------------------------------------------------------------

# 各處理階段的中文標籤（供心跳回報「目前卡在哪一步」）
PHASES = {
    "decode": "解碼", "detect": "偵測", "track": "追蹤補洞",
    "censor": "打碼", "write": "編碼寫入", "finalize": "收尾（等 ffmpeg 完成編碼）", "done": "完成",
}


class _Heartbeat:
    """背景執行緒：定期回報處理進度，並在長時間沒有進展時指出卡在哪個階段。

    stats 由管線各執行緒更新：frames_read（已解碼）、written（已輸出）、t_advance（最後一次
    有進展的時刻）、_q_dec / _q_out（管線佇列，滿了代表下游追不上，用來判斷瓶頸）。
    任一階段卡死（壞幀、DirectML 卡住、ffmpeg 沒反應）時 t_advance 停止推進，
    這個執行緒仍會印出「已 Xs 沒有進展，可能卡在【某階段】」與佇列狀態。
    """

    def __init__(self, log, stats: dict, total: int, interval: float = 15.0, stall_after: float = 20.0):
        self.log = log
        self.stats = stats
        self.total = total
        self.interval = interval
        self.stall_after = stall_after
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=3)
        return False

    def _stage_shares(self, elapsed: float) -> str:
        """累計各階段耗時佔比（取前三名、≥5% 才列），處理中就能看出瓶頸在哪個階段。"""
        pairs = [("解碼", "t_decode"), ("偵測", "t_detect"), ("追蹤", "t_track"),
                 ("打碼", "t_censor"), ("轉bytes", "t_tobytes"), ("送編碼", "t_pipe")]
        seg = sorted(((zh, self.stats.get(k, 0.0) / elapsed) for zh, k in pairs),
                     key=lambda p: p[1], reverse=True)
        top = [f"{zh} {v:.0%}" for zh, v in seg[:3] if v >= 0.05]
        return f"，累計 {'、'.join(top)}" if top else ""

    def _where(self) -> str:
        """判斷目前在忙／卡住的階段。管線模式看佇列狀態（滿＝下游追不上），否則退回 phase 標籤。"""
        phase = self.stats.get("phase")
        if phase in ("finalize", "done"):
            return PHASES[phase]
        qd, qo = self.stats.get("_q_dec"), self.stats.get("_q_out")
        if qd is None or qo is None:
            return PHASES.get(phase, phase or "?")
        if qo.full():
            return PHASES["write"]   # 輸出佇列滿：編碼寫入跟不上
        if qd.full():
            return PHASES["detect"]  # 解碼佇列滿：偵測跟不上
        return PHASES["decode"]      # 兩條都空：解碼供不上

    def _run(self):
        t_start = time.time()
        last_report = t_start
        last_frames = 0
        stalled = False
        while not self._stop.wait(3.0):
            now = time.time()
            frames = self.stats["frames_read"]
            written = self.stats.get("written", 0)
            where = self._where()
            since_advance = now - self.stats.get("t_advance", t_start)
            if since_advance < self.stall_after:
                stalled = False
                if now - last_report < self.interval:
                    continue
                el = now - t_start
                # last_frames / last_report 只在每次回報時一起更新，兩者的時間窗才會一致
                inst = (frames - last_frames) / (now - last_report)
                avg = frames / el if el else 0
                pct = f"{frames / self.total:.0%}" if self.total > 0 else f"{frames} 幀"
                eta = f"，剩約 {(self.total - frames) / inst:.0f}s" if self.total > 0 and inst > 0.1 else ""
                rss = process_rss_mb()
                extra = f"，本幀 {self.stats.get('last_boxes', 0)} 框"
                if self.stats.get("last_max_side"):
                    extra += f"（最大 {self.stats['last_max_side']}px）"
                extra += self._stage_shares(el)
                if rss:
                    extra += f"，記憶體 {rss:.0f}MB"
                self.log(f"⏳ 進度 {pct}（解碼 {frames}、輸出 {written} 幀，即時 {inst:.1f} fps、"
                         f"平均 {avg:.1f} fps，瓶頸【{where}】{extra}{eta}）")
            elif not stalled or now - last_report >= self.interval:
                # 一進入卡住狀態立刻警告，之後每 interval 秒重複一次（原本每 3 秒洗版）
                qd, qo = self.stats.get("_q_dec"), self.stats.get("_q_out")
                qinfo = (f"，佇列 {qd.qsize()}/{qd.maxsize}·{qo.qsize()}/{qo.maxsize}"
                         if qd is not None and qo is not None else "")
                self.log(f"⚠ 已 {since_advance:.0f} 秒沒有進展，可能卡在【{where}】"
                         f"（解碼 {frames}" + (f"/{self.total}" if self.total > 0 else "") +
                         f" 幀、輸出 {written} 幀{qinfo}）")
                stalled = True
            else:
                continue
            last_report = now
            last_frames = frames


def _open_video(path: Path, log):
    """開啟影片，優先用 FFmpeg 後端的硬體加速解碼（NVDEC / D3D11 / QSV，由 FFmpeg 自選）。

    先開一次並試讀第一幀確認可用，再重開一次從頭讀；試讀失敗（驅動或格式不支援）
    就退回預設開法。VIDEO_ACCELERATION_ANY 本身允許 FFmpeg 內部退回軟解，這裡的
    失敗處理只是保險。
    """
    if hasattr(cv2, "CAP_PROP_HW_ACCELERATION"):
        params = [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY]
        try:
            cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, params)
            ok = cap.isOpened() and cap.read()[0]
            cap.release()
            if ok:
                cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, params)
                if cap.isOpened():
                    log("解碼：FFmpeg 硬體加速優先（不支援的格式由 FFmpeg 自動退回軟解）")
                    return cap
                cap.release()
        except Exception:  # noqa: BLE001 — 任何失敗都退回預設開法
            pass
    return cv2.VideoCapture(str(path))


_RETRY = object()  # _process_video_once 的回傳哨兵：ffmpeg 編碼失敗，請改用 OpenCV 重跑


def _process_video_once(path, out_path, detector, args, sink: str, ffmpeg, log, progress, cancel):
    encoder = pick_encoder(getattr(args, "encoder", "auto")) if sink == "ffmpeg" else None

    cap = _open_video(path, log)
    if not cap.isOpened():
        log(f"⚠ 無法開啟影片：{path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 先印影片基本資訊，變慢時一眼看出是不是解析度太高（直式影片帶旋轉 metadata 時寬高可能對調）
    w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = f"（約 {total / fps:.0f}s）" if total > 0 and fps > 0 else ""
    log(f"▶ {path.name}：{w0}x{h0}，{fps:.0f} fps，共 {total} 幀{dur}")
    stats = {"frames_read": 0, "written": 0, "faces": 0, "covered": 0, "cancelled": False,
             "t_decode": 0.0, "t_detect": 0.0, "t_track": 0.0, "t_censor": 0.0,
             "t_tobytes": 0.0, "t_pipe": 0.0, "phase": None, "t_advance": time.time(),
             "last_boxes": 0, "max_boxes": 0, "last_max_side": 0}
    rescue_before = rescue_stats(detector)
    t_start = time.time()

    def report(done: int):
        if progress is not None:  # GUI 進度條；CLI 進度改由 _Heartbeat 統一回報
            progress(done, total)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 三段式管線：解碼 → 偵測+追蹤 → 打碼+編碼，各在自己的執行緒重疊執行 ---
    # cap.read / onnxruntime session.run / cv2 運算都會釋放 GIL，執行緒化是真併行；
    # 原本單執行緒串行時各階段互等，CPU 與 GPU 都吃不滿。佇列刻意小（4K 幀約 24MB），
    # 佔記憶體有限、滿了自然形成背壓；哪條佇列滿也讓心跳能指出瓶頸在哪一段。
    q_dec: queue.Queue = queue.Queue(maxsize=4)   # 解碼 → 偵測
    q_out: queue.Queue = queue.Queue(maxsize=4)   # 追蹤輸出（幀+框）→ 打碼+編碼
    stats["_q_dec"], stats["_q_out"] = q_dec, q_out
    stop = threading.Event()
    fail: dict[str, str] = {}
    writer_box: dict = {}

    def _put(q, item) -> bool:
        """帶背壓的 put：佇列滿就等，任一階段失敗（stop）立即放棄。"""
        while not stop.is_set():
            try:
                q.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def _get(q):
        while not stop.is_set():
            try:
                return q.get(timeout=0.2)
            except queue.Empty:
                continue
        return None  # 停止時視同資料結束

    def decode_loop():
        try:
            while True:
                if cancel is not None and cancel():
                    stats["cancelled"] = True
                    break
                t0 = time.time()
                ok, frame = cap.read()
                stats["t_decode"] += time.time() - t0
                if not ok:
                    break
                stats["frames_read"] += 1
                stats["t_advance"] = time.time()
                report(stats["frames_read"])
                if not _put(q_dec, frame):
                    return
        except Exception as e:  # noqa: BLE001
            fail["decode"] = f"解碼失敗：{e}"
            stop.set()
        finally:
            _put(q_dec, None)

    def detect_loop():
        """跑在呼叫者執行緒：偵測器與建立它的執行緒相同，GUI 跨輪沿用行為不變。"""
        tracker = StreamTracker(min_hits=getattr(args, "min_hits", 2)) if getattr(args, "track", True) else None
        sticky: list = []
        try:
            while True:
                frame = _get(q_dec)
                if frame is None:
                    break
                t0 = time.time()
                boxes = detector.detect(frame)
                stats["t_detect"] += time.time() - t0
                stats["faces"] += len(boxes)
                if tracker is not None:
                    t0 = time.time()
                    emitted = tracker.push(frame, boxes)
                    stats["t_track"] += time.time() - t0
                else:
                    # 逐幀模式：偵測框延續 keep 幀補空窗（原 process_frame 的 sticky 邏輯）
                    sticky[:] = [(b, ttl - 1) for b, ttl in sticky if ttl > 1]
                    sticky.extend((b, args.keep) for b in boxes)
                    emitted = [(frame, [b for b, _ in sticky])]
                for item in emitted:
                    if not _put(q_out, item):
                        return
            if tracker is not None and not stats["cancelled"] and not fail:
                for item in tracker.flush():
                    if not _put(q_out, item):
                        return
        except Exception as e:  # noqa: BLE001
            fail["detect"] = f"偵測失敗：{e}"
            stop.set()
        finally:
            _put(q_out, None)

    def write_loop():
        try:
            while True:
                item = _get(q_out)
                if item is None:
                    break
                frame, boxes = item
                stats["covered"] += len(boxes)
                stats["last_boxes"] = len(boxes)
                stats["last_max_side"] = max((max(b[2], b[3]) for b in boxes), default=0)
                stats["max_boxes"] = max(stats["max_boxes"], len(boxes))
                t0 = time.time()
                apply_boxes(frame, boxes, args)
                stats["t_censor"] += time.time() - t0
                if "writer" not in writer_box:
                    # 直式手機影片帶旋轉 metadata 時 CAP_PROP 的寬高可能與實際幀不符，以第一幀實際尺寸為準
                    h, w = frame.shape[:2]
                    writer_box["size"] = (w, h)
                    writer_box["writer"] = (
                        _FfmpegWriter(ffmpeg, out_path, path, fps, (w, h), encoder)
                        if sink == "ffmpeg" else _Cv2Writer(out_path, fps, (w, h))
                    )
                if sink == "ffmpeg":
                    t0 = time.time()
                    raw = frame.tobytes()                  # BGR → bytes：純 CPU / 記憶體頻寬
                    stats["t_tobytes"] += time.time() - t0
                    t0 = time.time()
                    writer_box["writer"].write_raw(raw)    # 卡在這裡代表編碼器跟不上（背壓）
                    stats["t_pipe"] += time.time() - t0
                else:
                    t0 = time.time()
                    writer_box["writer"].write(frame)
                    stats["t_pipe"] += time.time() - t0
                stats["written"] += 1
                stats["t_advance"] = time.time()
        except Exception as e:  # noqa: BLE001 — 常見為 ffmpeg 中途掛掉的 BrokenPipeError
            fail["write"] = str(e)
            stop.set()

    writer = None
    t_finalize = 0.0
    with _Heartbeat(log, stats, total):
        threads = [threading.Thread(target=decode_loop, daemon=True),
                   threading.Thread(target=write_loop, daemon=True)]
        for t in threads:
            t.start()
        detect_loop()
        for t in threads:
            t.join()
        cap.release()
        stats["phase"] = "finalize"
        writer = writer_box.get("writer")
        writer_err = None
        if writer is not None:
            _t = time.time()
            writer_err = writer.close(abort=stats["cancelled"] or bool(fail))
            t_finalize = time.time() - _t
    stats["phase"] = "done"
    written = stats["written"]
    error = writer_err or fail.get("write")
    elapsed = max(time.time() - t_start, 1e-9)
    stats["t_write"] = stats["t_tobytes"] + stats["t_pipe"]
    # 各階段「忙碌時間」佔整體耗時的比例。管線並行後各階段重疊執行，總和可以超過 100%；
    # 佔比最高的那項就是目前的瓶頸
    timed = [("解碼", "t_decode"), ("偵測", "t_detect"), ("追蹤", "t_track"), ("打碼", "t_censor"),
             ("轉bytes", "t_tobytes"), ("送編碼", "t_pipe")]
    seg = [(zh, stats[v]) for zh, v in timed] + [("收尾", t_finalize)]
    parts = ", ".join(f"{zh} {t / elapsed:.0%}" for zh, t in seg if t / elapsed >= 0.005)

    if stats["cancelled"]:
        out_path.unlink(missing_ok=True)
        log(f"⚠ 已取消：{path.name}（已解碼 {stats['frames_read']} 幀、輸出 {written} 幀，"
            f"{elapsed:.1f}s；忙碌佔比 {parts}）")
        return None
    if fail.get("decode") or fail.get("detect"):
        # 解碼/偵測層的錯誤換編碼器也救不回來，不走 ffmpeg 重試
        out_path.unlink(missing_ok=True)
        log(f"⚠ 無法處理 {path.name}：{fail.get('decode') or fail.get('detect')}")
        return None
    if error is not None:
        out_path.unlink(missing_ok=True)
        if sink == "ffmpeg":
            log(f"⚠ ffmpeg 編碼失敗：{error}")
            return _RETRY
        log(f"⚠ 無法輸出 {path.name}：{error}")
        return None
    if written == 0:
        out_path.unlink(missing_ok=True)
        log(f"⚠ 無法解碼任何畫面：{path.name}")
        return None

    if getattr(args, "track", True):
        detail = f"偵測 {stats['faces']} 次，追蹤補齊後遮蔽 {stats['covered']} 次"
    else:
        detail = f"累計偵測 {stats['faces']} 次人臉"
    rescue_after = rescue_stats(detector)
    if rescue_before and rescue_after and rescue_after[0] > rescue_before[0]:
        trig, rf, rb = (a - b for a, b in zip(rescue_after, rescue_before))
        detail += f"；旋轉補救觸發 {trig} 幀、救回 {rf} 幀 {rb} 框"
    avg_boxes = stats["covered"] / max(1, written)
    box_note = f"，平均 {avg_boxes:.1f} 框/幀（尖峰 {stats['max_boxes']}）" if avg_boxes >= 3 else ""
    w, h = writer_box["size"]
    log(f"✓ {path.name} → {out_path.name}（{w}x{h}，共 {written} 幀，{elapsed:.1f}s ≈ {written / elapsed:.1f} fps；"
        f"忙碌佔比 {parts}{box_note}；{detail}；編碼 {writer.encoder}）")
    return written, stats["faces"]


def process_video(path: Path, out_path: Path, detector, args, log=print, progress=None, cancel=None):
    """處理單支影片。回傳 (幀數, 累計偵測次數)；失敗或取消回傳 None。

    progress(done, total) 回報進度；cancel() 回傳 True 時中止並清理。
    影片只解碼一遍：追蹤模式（預設）用 StreamTracker 線上補洞，args.track=False 則逐幀即時處理。
    畫面直接經 stdin 送進 ffmpeg，一次完成編碼（依 args.encoder 可用硬體編碼器）與原始音軌合併，
    不再經過中間檔重複壓縮；沒有 ffmpeg 時退回 OpenCV 輸出（無音軌），ffmpeg 編碼失敗也會用 OpenCV 重跑。
    """
    ffmpeg = find_ffmpeg()
    for sink in (["ffmpeg", "cv2"] if ffmpeg else ["cv2"]):
        result = _process_video_once(path, out_path, detector, args, sink, ffmpeg, log, progress, cancel)
        if result is not _RETRY:
            return result
        log("  （改用 OpenCV 重新輸出，將不含音軌）")
    return None


def default_output(path: Path, out_dir: Path | None) -> Path:
    suffix = ".mp4" if path.suffix.lower() in VIDEO_EXTS else path.suffix
    name = f"{path.stem}_blurred{suffix}"
    return (out_dir / name) if out_dir else path.with_name(name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AI 人臉自動打碼工具（照片 / 影片）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="輸入的照片、影片，或整個資料夾")
    parser.add_argument("-o", "--output", help="輸出檔案或資料夾（預設在原檔旁加 _blurred）")
    parser.add_argument("--mode", choices=["mosaic", "blur"], default="mosaic",
                        help="打碼方式：mosaic 馬賽克 / blur 高斯模糊")
    parser.add_argument("--strength", type=int, default=5, choices=range(1, 11), metavar="1-10",
                        help="打碼強度，越大越模糊")
    parser.add_argument("--detector", choices=["scrfd", "yunet", "both"], default="scrfd",
                        help="偵測器：scrfd 高準確度 / yunet 輕量快速 / both 兩者聯集（最高召回）")
    parser.add_argument("--det-size", type=int, default=1280,
                        help="SCRFD 偵測解析度；小臉多用 1280-1920，追求速度用 640")
    parser.add_argument("--conf", type=float, default=0.4,
                        help="人臉偵測信心門檻（漏抓調低、誤抓調高）")
    parser.add_argument("--pad", type=float, default=0.15,
                        help="偵測框向外擴張比例")
    parser.add_argument("--keep", type=int, default=4,
                        help="影片中偵測框延續的幀數，用來補偵測空窗")
    parser.add_argument("--ellipse", action="store_true", help="使用橢圓形遮罩（預設矩形）")
    parser.add_argument("--head", action="store_true",
                        help="加上頭部偵測（含背對鏡頭、極端角度），遮蔽範圍從臉擴大到整顆頭")
    parser.add_argument("--head-conf", type=float, default=0.5,
                        help="頭部偵測門檻（與 --conf 獨立）；室內圓弧物體被誤當成頭時調高到 0.6")
    parser.add_argument("--no-track", dest="track", action="store_false",
                        help="停用影片追蹤補洞，改回逐幀即時處理")
    parser.add_argument("--min-hits", type=int, default=2,
                        help="影片中一條軌跡至少要被偵測到幾幀才輸出；只出現一幀的框多為誤判，設 1 停用過濾")
    parser.add_argument("--no-multiscale", dest="multiscale", action="store_false",
                        help="停用多尺度：預設 det-size 高於 640 時會再加一道 640 掃描取聯集，補特寫大臉")
    parser.add_argument("--rescue", action="store_true",
                        help="啟用旋轉補救：整幀沒抓到臉時，把畫面轉 90/270 度用臉部模型（不含頭部）再跑一次，"
                             "救回橫躺、大角度歪斜的臉；誤框率為 SCRFD 本身的水準，預設關")
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto",
                        help="偵測運算裝置：auto 有 GPU 就用（macOS CoreML / Windows DirectML）、"
                             "不行自動退回 CPU；gpu 強制用 GPU（不可用時報錯）")
    parser.add_argument("--encoder", choices=ENCODER_CHOICES, default="auto",
                        help="影片編碼器：auto 有硬體編碼器（VideoToolbox / NVENC / QSV / AMF）就用，"
                             "否則 libx264；software 一律 libx264；hardware 強制硬體（不可用時報錯）")
    args = parser.parse_args()

    in_path = Path(args.input).expanduser()
    if not in_path.exists():
        sys.exit(f"找不到輸入：{in_path}")

    print(runtime_info())
    detector = create_detector(args, log=print)
    print(f"偵測裝置：{device_label(detector)}")

    if in_path.is_dir():
        out_dir = Path(args.output).expanduser() if args.output else in_path / "blurred"
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(
            p for p in in_path.iterdir()
            if p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS and not p.name.endswith(f"_blurred{p.suffix}")
        )
        if not files:
            sys.exit("資料夾內沒有支援的照片或影片")
        print(f"共 {len(files)} 個檔案，輸出到 {out_dir}/")
        for p in files:
            out = default_output(p, out_dir)
            if p.suffix.lower() in VIDEO_EXTS:
                process_video(p, out, detector, args)
            else:
                process_image(p, out, detector, args)
    else:
        ext = in_path.suffix.lower()
        out = Path(args.output).expanduser() if args.output else default_output(in_path, None)
        out.parent.mkdir(parents=True, exist_ok=True)
        if ext in VIDEO_EXTS:
            process_video(in_path, out, detector, args)
        elif ext in IMAGE_EXTS:
            process_image(in_path, out, detector, args)
        else:
            sys.exit(f"不支援的檔案格式：{ext}")


if __name__ == "__main__":
    main()
