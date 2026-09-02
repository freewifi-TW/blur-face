# blur-face — AI 人臉自動打碼工具

自動偵測照片、影片中的人臉並打上馬賽克或高斯模糊。**全程在本機離線處理**，檔案不會上傳到任何地方，也不消耗任何 AI token。

人臉偵測預設使用 **SCRFD-10G**（InsightFace，WIDER FACE hard 集約 83% AP），對**側臉、小臉、遮擋臉**都有高召回率；另內建輕量的 YuNet 可選。

進階遮蔽能力：

- **頭部偵測**（`--head` / GUI 勾選）：加掛 CrowdHuman YOLOv5m 頭部模型，**背對鏡頭、極端角度的人也會被遮蔽**（從「遮臉」升級為「遮頭」）。門檻 `--head-conf` 與人臉獨立，室內門把、燈罩、椅背被誤當成頭時調高到 0.6
- **影片追蹤補洞**（預設開啟，`--no-track` 停用）：邊偵測邊把同一個人串成時間軸軌跡，短暫漏偵測的幀用前後幀線性內插補齊，並向軌跡前後延伸，消除馬賽克閃爍與漏幀；線上演算法只需暫存二十幾幀，影片只解碼一遍。**只出現一幀的框視為誤判不輸出**（`--min-hits`，預設 2），避免單幀誤判被延伸成 13 幀的馬賽克塊
- **多尺度偵測**（預設開啟，`--no-multiscale` 停用）：det-size 高於 640 時自動多跑一道 640 掃描取聯集。SCRFD 的訓練畫布是 640，特寫大臉在高解析度掃描下會超出模型見過的尺度而漏掉，640 那道把它補回來；GPU 每幀只多約 14 ms
- **特寫／旋轉補救**（預設關閉，`--rescue` / GUI 勾選啟用）：整幀沒抓到臉時，用 YuNet 對轉正（0 / 90 / 270 度）且縮小一半的畫面再試一次，專救「手機貼臉自拍、畫面橫躺、臉被裁掉一半」的極端特寫。一般拍別人的影片不建議開：人物背對或低頭的瞬間，它可能把頭髮、衣服框成臉
- **GPU 加速**（預設自動，`--device` 控制）：偵測在 macOS 走 CoreML、Windows 走 DirectML（NVIDIA / AMD / Intel 內顯皆可，不需裝 CUDA），M4 實測偵測快 4 倍、整體快 3.4 倍，結果與 CPU 完全相同；沒有可用 GPU 自動退回 CPU
- **硬體編碼**（預設自動，`--encoder` 控制）：影片畫面直接串流進 ffmpeg 一次完成編碼與音軌合併，不再經過中間檔重複壓縮；可用時改用 VideoToolbox / NVENC / QSV / AMF 硬體編碼器

提供三種使用方式：**桌面 App（GUI）**、**命令列（CLI）**、Python 模組。

## 桌面 App

```bash
# 開發模式直接執行
.venv/bin/python gui.py
```

拖放檔案或資料夾 → 調整設定 → 開始處理。支援批次、即時進度、中途取消。「GPU 加速偵測」與「硬體編碼影片」預設開啟，開始處理後狀態列會顯示實際使用的裝置與編碼器。

下方「處理紀錄」面板會記錄執行環境（onnxruntime provider、顯示卡型號與驅動）、每個模型實際落在 GPU 或 CPU 與失敗原因、每個檔案完整的階段佔比（解碼 / 偵測 / 追蹤 / 打碼 / 轉bytes / 送編碼 / 收尾，加總約 100%）。處理影片時每隔約 15 秒回報一次即時進度與 fps；**畫面若卡住不動，會標出「已 N 秒沒有新畫面，可能卡在【某階段】」，一眼看出是解碼、偵測還是編碼卡住。處理變慢或懷疑沒用到 GPU 時，按「複製紀錄」貼出來就能判斷。** 偵測設定沒變時會沿用同一個偵測器，不會每輪重建 GPU session。

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
- GPU 加速不用另外裝：macOS 用系統內建 CoreML；Windows 由 `requirements.txt` 依平台自動改裝 `onnxruntime-directml`，任何支援 DirectX 12 的顯示卡（含內顯）都能用

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

`models/crowdhuman_yolov5m.onnx`（頭部偵測）沒有現成下載點：它是從
[MK-CUPIST/crowdhuman_yolov5m](https://huggingface.co/MK-CUPIST/crowdhuman_yolov5m) 的 `.pt`
用 yolov5 v5.0 的 `models/export.py --grid` 轉出的（本 repo 已內含轉好的檔案）。

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

# 極端場景追求最高召回：雙偵測器聯集 + 頭部偵測 + 低門檻 + 高偵測解析度
.venv/bin/python blur_faces.py photo.jpg --detector both --head --conf 0.3 --det-size 1920

# 強制只用 CPU 偵測、libx264 軟體編碼（例如要排除 GPU 因素，或追求最高畫質）
.venv/bin/python blur_faces.py video.mp4 --device cpu --encoder software
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
| `--head` | 關 | 加上頭部偵測（含背對鏡頭），遮蔽範圍從臉擴大到整顆頭 |
| `--head-conf` | `0.5` | 頭部偵測門檻，與 `--conf` 獨立；室內圓弧物體被誤當成頭時調到 0.6 |
| `--no-track` | （追蹤預設開） | 停用影片追蹤補洞，改回逐幀即時處理 |
| `--min-hits` | `2` | 影片中一條軌跡至少要被偵測到幾幀才輸出；設 1 停用（保留單幀框） |
| `--no-multiscale` | （預設開） | 停用多尺度：預設 det-size 高於 640 時再加一道 640 掃描，補特寫大臉 |
| `--rescue` | 關 | 啟用特寫／旋轉補救：整幀沒抓到時用 YuNet 對 0/90/270 度、縮小畫布再試；只建議用於貼臉自拍類特寫 |
| `--device` | `auto` | 偵測運算裝置：`auto` 有 GPU 就用、不行退回 CPU / `cpu` / `gpu`（不可用時報錯） |
| `--encoder` | `auto` | 影片編碼器：`auto` 有硬體編碼器就用、否則 libx264 / `software` 一律 libx264 / `hardware`（不可用時報錯） |
| `--keep` | `4` | 逐幀模式下偵測框延續幀數（追蹤模式不使用） |
| `--ellipse` | 關 | 橢圓形遮罩（預設矩形） |

## 準確度實測（29 人合照，M 系列 Mac）

| 情境 | YuNet | SCRFD（預設） |
|---|---|---|
| 原圖 3000px | 29/29 | 29/29 |
| 小臉（縮到 700px，臉約 25px） | 29/29 + 1 誤報 | 29/29 |
| 遮擋（下半臉全遮） | 28/29 | 29/29 |
| 旋轉 25 度 | 25/29 | 29/29 |
| 側臉（2 張真實側臉照） | — | 全中 |
| 背對鏡頭人群（GT 15+ 顆頭） | 2（純臉部） | **21**（開 `--head`） |

### 側臉與特寫實測

9 張 Wikimedia Commons 側臉照片、3 張使用者提供的極端特寫（臉比畫面大、橫躺 90 度、部分裁切、動態模糊）、6 張無人照片，以及把 3 張特寫合成的 90 幀影片：

| 設定 | 側臉 9 張 | 極端特寫 3 張 | 特寫影片 90 幀 | 無人照片誤報 |
|---|---|---|---|---|
| SCRFD 1280 + 頭部（v1.2.1 預設） | 5 | 0 | 0 | — |
| 加多尺度 | 8 | 0 | 0 | 不變 |
| 加多尺度 + 特寫補救（`--rescue`） | 8 | 3 | 90 | 補救 0 框 |

還漏的一張是第一人稱只看到耳朵與太陽眼鏡側面的極端照。中等尺寸的側臉 SCRFD 本來就抓得到（分數 0.5 到 0.8），漏的多是特寫大臉，這是尺度問題不是角度問題。

誤框方面，17 張無人照片在門檻 0.5 下 SCRFD 框出的 18 個「臉」有 16 個是牆上相框裡的真人臉，頭部模型框到的 4 個有 2 個是遠處登山客的頭；真正的誤框是教堂長椅的圓弧椅端（頭部 0.61 / 0.54）、一塊帆布與一片屋頂（SCRFD 0.51 / 0.52）。補救模式若在「有人但那一瞬間臉和頭都沒抓到」的幀啟動，約四成會多框到頭髮、衣服，因此改為預設關閉。

### 速度（Apple M4）

單幀偵測（含前處理，1080p 輸入）：

| 模型 | CPU | GPU（CoreML） |
|---|---|---|
| SCRFD `--det-size 1280` | 234 ms | 60 ms |
| SCRFD `--det-size 640` | 64 ms | 14 ms |
| 頭部 YOLOv5m（`--head`） | 105 ms | 30 ms |
| YuNet | 18 ms | —（OpenCV DNN 只跑 CPU） |

12 秒 720p 影片端對端（`--head`，預設 1280）：

| 設定 | 耗時 |
|---|---|
| 舊版（CPU、影片解碼兩遍、mp4v + libx264 雙重編碼） | 128 s |
| `--device cpu --encoder software` | 126 s |
| `--device auto --encoder software` | 42 s |
| `--device auto --encoder auto`（預設） | 37 s |

四種設定的偵測次數與遮蔽次數完全相同。偵測仍是主要瓶頸，硬體編碼在 720p 差異不大，4K 才明顯。

## 注意事項

- 嚴重模糊、極小的臉仍可能漏偵測。機敏內容建議開啟 `--head`（連背對鏡頭的頭都遮），並用 `--detector both --conf 0.3` 拉滿召回率，**發布前抽查輸出結果**。
- 馬賽克 / 模糊在理論上有被還原的風險，對機敏內容可搭配較高 `--strength`。
- 硬體編碼器（預設 `auto`）速度快，但同檔案大小下畫質略低於 libx264，且各家畫質參數不完全對齊；對畫質有要求可用 `--encoder software`。
