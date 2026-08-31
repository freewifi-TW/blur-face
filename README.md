# blur-face — AI 人臉自動打碼工具

自動偵測照片、影片中的人臉並打上馬賽克或高斯模糊。**全程在本機離線處理**，檔案不會上傳到任何地方，也不消耗任何 AI token。

人臉偵測預設使用 **SCRFD-10G**（InsightFace，WIDER FACE hard 集約 83% AP），對**側臉、小臉、遮擋臉**都有高召回率；另內建輕量的 YuNet 可選。

提供三種使用方式：**桌面 App（GUI）**、**命令列（CLI）**、Python 模組。

## 桌面 App

```bash
# 開發模式直接執行
.venv/bin/python gui.py
```

拖放檔案或資料夾 → 調整設定 → 開始處理。支援批次、即時進度、中途取消。

### 打包成應用程式

PyInstaller 無法跨平台編譯，macOS 版要在 Mac 上建、Windows 版要在 Windows 上建：

```bash
# 本機打包（在 macOS 上產出 dist/BlurFace.app；Windows 上把 : 換成 ; 產出 dist/BlurFace/BlurFace.exe）
.venv/bin/pip install pyinstaller
.venv/bin/pyinstaller --noconfirm --windowed --name BlurFace \
  --icon assets/icon.icns --add-data "models:models" gui.py
```

或推上 GitHub 用附的 CI 自動建雙平台：`.github/workflows/build.yml` 會在打 `v*` tag（或手動觸發 workflow_dispatch）時同時產出 `BlurFace-macos.zip` 和 `BlurFace-windows.zip`（在 Actions 的 artifacts 下載）。

未簽章 app 的注意事項：

- **macOS**：第一次開啟要對 app 右鍵 →「打開」，或執行 `xattr -cr BlurFace.app` 解除隔離
- **Windows**：SmartScreen 會警告，點「其他資訊」→「仍要執行」

## 環境需求（開發模式）

- Python 3.10+（已在 3.14 測試）
- ffmpeg 不用另外裝（內建 imageio-ffmpeg 靜態版；系統有裝會優先用系統版）

## 安裝

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

模型檔已放在 `models/`。若需要重新下載：

```bash
# SCRFD-10G（預設，高準確度）
curl -L -o models/det_10g.onnx \
  "https://huggingface.co/deepghs/insightface/resolve/main/buffalo_l/det_10g.onnx"

# YuNet（輕量備用）
curl -L -o models/face_detection_yunet_2023mar.onnx \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
```

## 使用方式

```bash
# 照片（預設馬賽克，輸出到 photo_blurred.jpg）
.venv/bin/python blur_faces.py photo.jpg

# 影片（自動保留音軌，輸出 H.264 mp4）
.venv/bin/python blur_faces.py video.mp4

# 整個資料夾批次處理（輸出到 <資料夾>/blurred/）
.venv/bin/python blur_faces.py ~/Pictures/trip/

# 高斯模糊 + 橢圓遮罩，強度 8
.venv/bin/python blur_faces.py photo.jpg --mode blur --ellipse --strength 8

# 先預覽偵測框（不打碼），確認有沒有漏抓
.venv/bin/python blur_faces.py photo.jpg --preview

# 極端場景追求最高召回：雙偵測器聯集 + 低門檻 + 高偵測解析度
.venv/bin/python blur_faces.py photo.jpg --detector both --conf 0.3 --det-size 1920
```

## 參數

| 參數 | 預設 | 說明 |
|---|---|---|
| `-o, --output` | 原檔旁加 `_blurred` | 輸出檔案或資料夾 |
| `--mode` | `mosaic` | `mosaic` 馬賽克 / `blur` 高斯模糊 |
| `--strength` | `5` | 打碼強度 1–10，越大越模糊 |
| `--detector` | `scrfd` | `scrfd` 高準確度 / `yunet` 輕量快速 / `both` 聯集（最高召回） |
| `--det-size` | `1280` | SCRFD 偵測解析度；小臉多可調到 1920，追求速度用 640 |
| `--conf` | `0.4` | 偵測信心門檻；漏抓調低（如 0.3）、誤抓調高 |
| `--pad` | `0.15` | 偵測框向外擴張比例 |
| `--keep` | `4` | 影片中偵測框延續幀數，補偵測空窗、減少閃爍 |
| `--ellipse` | 關 | 橢圓形遮罩（預設矩形） |
| `--preview` | 關 | 只畫偵測框不打碼，用來檢查偵測效果 |

## 準確度實測（29 人合照，M 系列 Mac）

| 情境 | YuNet | SCRFD（預設） |
|---|---|---|
| 原圖 3000px | 29/29 | 29/29 |
| 小臉（縮到 700px，臉約 25px） | 29/29 + 1 誤報 | 29/29 |
| 遮擋（下半臉全遮） | 28/29 | 29/29 |
| 旋轉 25 度 | 25/29 | 29/29 |
| 側臉（2 張真實側臉照） | — | 全中 |

速度（單幀偵測）：SCRFD `--det-size 1280` 約 275ms、`640` 約 76ms；YuNet 約 15ms。

## 注意事項

- 極端角度（接近後腦勺）、嚴重模糊的臉仍可能漏偵測。**發布前建議先用 `--preview` 檢查**，機敏內容可用 `--detector both --conf 0.3` 拉滿召回率。
- 馬賽克 / 模糊在理論上有被還原的風險，對機敏內容可搭配較高 `--strength`。
