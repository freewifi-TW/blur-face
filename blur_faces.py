#!/usr/bin/env python3
"""AI 人臉自動打碼工具：支援照片與影片，偵測到的人臉自動打上馬賽克或高斯模糊。

人臉偵測預設使用 SCRFD-10G（InsightFace）模型，對側臉、小臉、遮擋臉有高召回率；
另提供輕量的 YuNet 與兩者聯集（both）模式。全程離線處理，檔案不會上傳。
影片處理完成後若系統有 ffmpeg，會自動把原始音軌接回輸出檔。
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

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


MODELS_DIR = resource_path("models")
SCRFD_MODEL = MODELS_DIR / "det_10g.onnx"
YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
HEAD_MODEL = MODELS_DIR / "crowdhuman_yolov5m.onnx"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


class ScrfdDetector:
    """SCRFD-10G 偵測器（onnxruntime），高召回率，對側臉/小臉/遮擋臉表現好。"""

    STRIDES = (8, 16, 32)
    NUM_ANCHORS = 2
    NMS_IOU = 0.4

    def __init__(self, conf: float, det_size: int):
        if not SCRFD_MODEL.exists():
            sys.exit(f"找不到模型檔 {SCRFD_MODEL}，請先下載（見 README.md）")
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.log_severity_level = 3  # 關閉動態輸入尺寸造成的無害 shape 警告
        self.session = ort.InferenceSession(
            str(SCRFD_MODEL), sess_options=so, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.conf = conf
        self.det_size = max(64, (det_size + 31) // 32 * 32)  # 需為 32 的倍數
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
    """YuNet 偵測器（OpenCV 內建），輕量快速。"""

    MAX_SIDE = 1280  # 偵測時長邊縮到此尺寸以內

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

    def __init__(self, conf: float):
        if not HEAD_MODEL.exists():
            sys.exit(f"找不到模型檔 {HEAD_MODEL}，請先下載（見 README.md）")
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(HEAD_MODEL), sess_options=so, providers=["CPUExecutionProvider"]
        )
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

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        boxes = [b for d in self.detectors for b in d.detect(frame)]
        if not boxes:
            return []
        # 用 NMS 去除高度重疊的重複框，分數一律 1.0（只做去重）
        idxs = cv2.dnn.NMSBoxes([list(map(float, b)) for b in boxes], [1.0] * len(boxes), 0.0, 0.5)
        return [boxes[i] for i in np.array(idxs).flatten()]


def create_detector(args):
    if args.detector == "scrfd":
        det = ScrfdDetector(args.conf, args.det_size)
    elif args.detector == "yunet":
        det = YunetDetector(args.conf)
    else:
        det = UnionDetector([ScrfdDetector(args.conf, args.det_size), YunetDetector(args.conf)])
    if getattr(args, "head", False):
        det = UnionDetector([det, HeadDetector(max(args.conf, 0.35))])
    return det


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
        k = max(5, (max(roi.shape[:2]) // (12 - strength)) | 1)  # 奇數 kernel
        censored = cv2.GaussianBlur(roi, (k, k), 0)

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


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def build_tracks(detections, iou_thresh=0.3, max_gap=15, extend=6, n_frames=None):
    """把逐幀偵測框串成軌跡：短暫漏偵測的幀用前後幀線性內插補齊，並向軌跡前後各延伸幾幀。

    detections: list[list[box]]，索引即幀號。回傳 dict[幀號] -> [box, ...]。
    """
    tracks: list[dict] = []  # {"boxes": {幀號: box}, "last": 最後出現的幀號}
    for idx, boxes in enumerate(detections):
        active = [t for t in tracks if idx - t["last"] <= max_gap]
        pairs = sorted(
            ((_iou(t["boxes"][t["last"]], b), ti, bi)
             for ti, t in enumerate(active) for bi, b in enumerate(boxes)),
            reverse=True,
        )
        used_t, used_b = set(), set()
        for iou_v, ti, bi in pairs:
            if iou_v < iou_thresh:
                break
            if ti in used_t or bi in used_b:
                continue
            active[ti]["boxes"][idx] = boxes[bi]
            active[ti]["last"] = idx
            used_t.add(ti)
            used_b.add(bi)
        for bi, b in enumerate(boxes):
            if bi not in used_b:
                tracks.append({"boxes": {idx: b}, "last": idx})

    n = n_frames if n_frames is not None else len(detections)
    per_frame: dict[int, list] = {}
    for t in tracks:
        idxs = sorted(t["boxes"])
        filled: dict[int, tuple] = {}
        for a, b in zip(idxs, idxs[1:]):
            filled[a] = t["boxes"][a]
            for i in range(a + 1, b):  # 漏偵測的幀：線性內插
                w = (i - a) / (b - a)
                filled[i] = tuple(
                    int(round(pa * (1 - w) + pb * w))
                    for pa, pb in zip(t["boxes"][a], t["boxes"][b])
                )
        filled[idxs[-1]] = t["boxes"][idxs[-1]]
        for i in range(max(0, idxs[0] - extend), idxs[0]):  # 軌跡起點往前延伸
            filled[i] = t["boxes"][idxs[0]]
        for i in range(idxs[-1] + 1, min(n, idxs[-1] + 1 + extend)):  # 終點往後延伸
            filled[i] = t["boxes"][idxs[-1]]
        for i, b in filled.items():
            per_frame.setdefault(i, []).append(b)
    return per_frame


def process_image(path: Path, out_path: Path, detector, args, log=print):
    img = cv2.imread(str(path))
    if img is None:
        log(f"⚠ 無法讀取圖片：{path}")
        return None
    n = process_frame(img, detector, args)
    cv2.imwrite(str(out_path), img)
    log(f"✓ {path.name} → {out_path.name}（偵測到 {n} 張人臉）")
    return n


def mux_audio(original: Path, video_only: Path, out_path: Path) -> bool:
    """用 ffmpeg 把原始檔的音軌接回處理後的影片，並轉成 H.264。"""
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        return False
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(video_only), "-i", str(original),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        str(out_path),
    ]
    kwargs = {}
    if sys.platform == "win32":
        # Windows 的 GUI 程式啟動 console 子程序會閃出主控台視窗，加旗標隱藏
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return subprocess.run(cmd, **kwargs).returncode == 0


def process_video(path: Path, out_path: Path, detector, args, log=print, progress=None, cancel=None):
    """progress(done, total) 回報進度；cancel() 回傳 True 時中止並清理。

    預設走兩段式（全片偵測 → 追蹤補洞 → 套用輸出）；args.track=False 時走
    逐幀即時處理（sticky_boxes 補償）。
    """
    if getattr(args, "track", True):
        return _process_video_tracked(path, out_path, detector, args, log, progress, cancel)
    return _process_video_stream(path, out_path, detector, args, log, progress, cancel)


def _process_video_tracked(path, out_path, detector, args, log, progress, cancel):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        log(f"⚠ 無法開啟影片：{path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 第一遍：全片偵測（最耗時），只收集偵測框
    detections: list[list] = []
    try:
        while True:
            if cancel is not None and cancel():
                log(f"⚠ 已取消：{path.name}")
                return None
            ok, frame = cap.read()
            if not ok:
                break
            detections.append(detector.detect(frame))
            i = len(detections)
            if progress is not None:
                progress(i, max(total, i) * 2)
            elif i % 100 == 0:
                pct = f"{i / total:.0%}" if total > 0 else f"{i} 幀"
                print(f"  {path.name}: 偵測中 {pct}", flush=True)
    finally:
        cap.release()

    n = len(detections)
    if n == 0:
        log(f"⚠ 無法解碼任何畫面：{path.name}")
        return None

    per_frame = build_tracks(detections, n_frames=n)
    total_faces = sum(len(b) for b in detections)
    covered = sum(len(v) for v in per_frame.values())

    # 第二遍：把追蹤補齊後的框套用到每一幀並輸出
    cap = cv2.VideoCapture(str(path))
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    writer = None
    idx = 0
    cancelled = False
    try:
        while True:
            if cancel is not None and cancel():
                cancelled = True
                break
            ok, frame = cap.read()
            if not ok:
                break
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            apply_boxes(frame, per_frame.get(idx, []), args)
            writer.write(frame)
            idx += 1
            if progress is not None:
                progress(n + idx, n * 2)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if cancelled or idx == 0:
        tmp_path.unlink(missing_ok=True)
        log(f"⚠ {'已取消' if cancelled else '無法輸出任何畫面'}：{path.name}")
        return None

    if mux_audio(path, tmp_path, out_path):
        tmp_path.unlink(missing_ok=True)
    else:
        shutil.move(str(tmp_path), str(out_path))
        log("  （未偵測到 ffmpeg 或合併失敗，輸出不含音軌）")
    log(f"✓ {path.name} → {out_path.name}（共 {idx} 幀，偵測 {total_faces} 次，追蹤補齊後遮蔽 {covered} 次）")
    return idx, total_faces


def _process_video_stream(path, out_path, detector, args, log, progress, cancel):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        log(f"⚠ 無法開啟影片：{path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    # 直式手機影片帶旋轉 metadata 時，CAP_PROP 回報的寬高可能與實際幀不符，
    # 因此以第一幀的實際尺寸初始化 writer
    writer = None

    sticky_boxes: list = []
    frame_idx = 0
    total_faces = 0
    cancelled = False
    try:
        while True:
            if cancel is not None and cancel():
                cancelled = True
                break
            ok, frame = cap.read()
            if not ok:
                break
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            total_faces += process_frame(frame, detector, args, sticky_boxes)
            writer.write(frame)
            frame_idx += 1
            if progress is not None:
                progress(frame_idx, total)
            elif frame_idx % 100 == 0:
                pct = f"{frame_idx / total:.0%}" if total > 0 else f"{frame_idx} 幀"
                print(f"  {path.name}: 已處理 {pct}", flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if cancelled or frame_idx == 0:
        tmp_path.unlink(missing_ok=True)
        log(f"⚠ {'已取消' if cancelled else '無法解碼任何畫面'}：{path.name}")
        return None

    if mux_audio(path, tmp_path, out_path):
        tmp_path.unlink(missing_ok=True)
    else:
        shutil.move(str(tmp_path), str(out_path))
        log("  （未偵測到 ffmpeg 或合併失敗，輸出不含音軌）")
    log(f"✓ {path.name} → {out_path.name}（共 {frame_idx} 幀，累計偵測 {total_faces} 次人臉）")
    return frame_idx, total_faces


def default_output(path: Path, out_dir: Path | None) -> Path:
    suffix = ".mp4" if path.suffix.lower() in VIDEO_EXTS else path.suffix
    name = f"{path.stem}_blurred{suffix}"
    return (out_dir / name) if out_dir else path.with_name(name)


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
    parser.add_argument("--no-track", dest="track", action="store_false",
                        help="停用影片兩段式追蹤補洞，改回逐幀即時處理（較省記憶體）")
    args = parser.parse_args()

    in_path = Path(args.input).expanduser()
    if not in_path.exists():
        sys.exit(f"找不到輸入：{in_path}")

    detector = create_detector(args)

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
