# Overwatch → DG-LAB | 守望先锋 → 郊狼

> 实时 OCR 识别游戏血量，掉血/死亡时触发郊狼波形输出
>
> Real-time OCR for in-game HP, triggering DG-LAB waveforms on damage and death

---

## 这是什么 / What Is This

打守望先锋时，OBS 推流血条区域到本机，程序实时 OCR 识别血量变化：

- **掉血** → 郊狼输出电击波形，血越少越强
- **死亡** → 全力输出死亡波形，12 秒内禁用
- **回血** → 自动停止波形
- **满血** → 波形停止，强度归零

所有参数（强度上限、波形样式、冷却时间）均可通过配置文件调整。

---

## 你需要准备 / Prerequisites

| 项目 | 说明 |
|------|------|
| 一台电脑运行 OBS | 需要能推流 MJPEG/UDP |
| 一台电脑运行本程序 | 可以和 OBS 同一台 |
| 手机安装 DG-LAB APP | [iOS](https://apps.apple.com/app/dg-lab/id6450055108) / [Android](https://play.google.com/store/apps/details?id=com.dg.lab.app) |
| 郊狼设备 | DG-LAB 3.0 / 2.0 均可 |
| 守望先锋 | 需要血条区域可见 |

> 本程序和 OBS 可以运行在同一台电脑上，也可以分开运行（手机和本程序需在同一局域网）。

---

## 快速开始 / Quick Start

### 第 1 步：克隆 & 安装依赖

```bash
git clone https://github.com/Stardust-minus/overwatch-dglab.git
cd overwatch-dglab

python3 -m venv venv
source venv/bin/activate
pip install openvino rapidocr-onnxruntime opencv-python pydglab-ws qrcode pyyaml
```

### 第 2 步：启动程序

```bash
# 同一台电脑运行 OBS 和本程序 (默认配置)
python main.py

# OBS 在另一台电脑 (需指定本机局域网 IP，让手机能扫码连接)
python main.py --ws ws://192.168.x.x:5678
```

启动后终端会显示一个二维码：

```
  请用 DG-LAB APP 扫描二维码绑定郊狼:

  ███████████████████████
  ████ ▄▄▄ █▀▄█ ▀▀▄ ████
  ████ █▀▀▀ █▄▀█ ▀▀▀ ████
  ...
```

### 第 3 步：手机扫码绑定

1. 打开手机 DG-LAB APP
2. 扫描终端中显示的二维码
3. 看到终端输出 `绑定成功!` 即可

### 第 4 步：OBS 设置推流

在 OBS 中配置：

1. **Settings → Output → Output Mode** → `Advanced`
2. **Recording → Type** → `Custom Output (FFmpeg)`
3. **Container Format** → `mjpeg`
4. **Video Encoder** → `mjpeg`
5. **Output URL** → `udp://127.0.0.1:1234`（同机）或 `udp://<接收端IP>:1234`（不同机器）
6. **Settings → Video → Output Resolution** → `256 x 64`
7. 拖动 OBS 输出窗口，**刚好框住血量数字部分**（如 `307/350`）

> 分辨率必须是 **256x64**，这是 OCR 优化的关键参数。输出窗口要尽量只包含血量数字，避免多余画面干扰识别。

### 第 5 步：开始游戏

OBS 开始推流后，程序会自动识别血量并触发波形。看到类似输出说明一切正常：

```
  Frame 1: ocr=2.1ms(avg=2.1ms) HP=350/350 [high]
  Frame 2: ocr=1.9ms(avg=2.0ms) HP=312/350 [high]
  [DAMAGE] HP=312/350 (89%) -> strength=11, pinch
  Frame 3: ocr=2.0ms(avg=2.0ms) HP=280/350 [high]
```

---

## 配置 / Configuration

编辑 `config/ow_blood_pulse.yaml`，所有参数都可调整：

### 安全 & 强度

```yaml
max_strength: 30    # 绝对硬上限，代码不可超越
strength_min: 8     # 满血时掉血强度
strength_max: 30    # 空血时掉血强度
```

掉血强度按血量比例线性映射：满血时最弱，空血时最强。死亡强度固定为 `max_strength`。

### 模式开关

```yaml
death_only: false     # true = 仅死亡时触发波形，掉血/回血/满血均不触发
```

或命令行直接启用：

```bash
python main.py --death-only
```

适合只想在死亡时才有反馈的场景。

### 时间参数

```yaml
damage_cooldown: 0.8    # 掉血事件冷却 (秒)
respawn_seconds: 12.0   # 死亡后禁用时长 (秒)
low_hp_ratio: 0.3       # 低于 max 的 30% 视为低血量
```

- `damage_cooldown`：掉血后多长时间内不再触发新波形，防止连续掉血时波形叠加
- `respawn_seconds`：死亡后忽略所有 OCR 读取，避免复活前乱触波形

### 波形选择

```yaml
damage_wave: "pinch"        # 掉血波形
low_hp_wave: "heartbeat"    # 低血量波形 (HP < max x 30%)
death_wave: "death_shock"   # 死亡波形
```

只需改名字即可切换波形。可用波形列表：

| 波形名 | 类型 | 时长 | 感受描述 |
|--------|------|------|----------|
| `pinch` | 掉血 | 700ms | 快速按捏，逐次减弱 |
| `sharp` | 掉血 | 300ms | 单次锐击脉冲 |
| `rapid` | 掉血 | 300ms | 快速三连击 |
| `heartbeat` | 低血 | 400ms | 心跳起伏节奏 |
| `slow_pulse` | 低血 | 500ms | 长间隔慢脉冲警告 |
| `death_shock` | 死亡 | 3s | 高频持续 + 衰减 |
| `death_crush` | 死亡 | 1.8s | 高频持续不减 |
| `death_fade` | 死亡 | 2s | 从强到弱缓慢消退 |

### 自定义波形

在 `waves:` 下添加新波形即可：

```yaml
waves:
  my_wave:
    desc: "我的自定义波形"
    data:
      - [[10, 10, 10, 10], [0, 0, 0, 0]]     # 静默 100ms
      - [[80, 80, 80, 80], [50, 80, 100, 80]] # 脉冲 100ms
    repeat: 2
```

格式说明：
- 每行 = 100ms
- `[频率x4]`：频率范围 10~240
- `[强度x4]`：强度范围 0~100（相对于 `set_strength` 的百分比）

---

## 事件逻辑 / Event Logic

```
首帧           -> 记录基线，不触发任何事件
HP 下降 >= 1   -> DAMAGED  -> 输出掉血波形，强度按比例，0.8s 冷却
HP 下降 + 低血 -> DAMAGED  -> 切换为低血量波形
HP 上升        -> HEALED   -> 停止波形输出
HP = max       -> FULL     -> 停止波形 + 强度归零
HP = 0         -> DEATH    -> 全力输出死亡波形，12s 内禁用
```

### 队列控制

每次事件触发时自动清空设备波形队列，防止波形叠加积累。队列深度实时跟踪，超限自动清空。

---

## 调试 & 测试 / Debug & Test

### 仅 OCR + 血量显示（不连接郊狼）

```bash
cd tests
python ocr_receiver_mjpeg.py --port 1234
```

### 本地模拟 OBS 推流

```bash
cd tests

# 发送随机文字测试帧
python udp_sender.py --random --port 1234

# 发送静态图片（循环）
python udp_sender.py --image test.png --port 1234
```

---

## 架构 / Architecture

```
OBS (远端/本机)                      本机
+--------------+   MJPEG/UDP   +---------------------------------+
|  256x64 区域  | ----------> | MJPEGReceiver                    |
|   60 FPS      |             |  | JPEG 帧边界 (FFD8/FFD9) 重组   |
|   Custom FFmpeg|            | cv2.imdecode -> BGR frame          |
+--------------+              |  |                               |
                              | OpenVINO FP32 CPU OCR (~2ms)     |
                              |  |                               |
                              | HPParser (锁定 max, 去 OCR 扰动)  |
                              |  |                               |
                              | BloodPulseController              |
                              |  | 事件 -> 波形 + 强度计算         |
                              |  |                               |
                              | 郊狼 DG-LAB WebSocket 输出        |
                              +---------------------------------+
```

---

## 项目结构 / Project Structure

```
overwatch-dglab/
+-- main.py                       # 主程序入口
+-- config/
|   +-- ow_blood_pulse.yaml       # 配置文件
+-- lib/
|   +-- hp_parser.py              # HP 解析器 (锁定 max, 去 OCR 扰动)
|   +-- rapidocr_openvino.py      # OpenVINO 推理后端适配
+-- models/
|   +-- fp32/                     # FP32 OpenVINO IR 模型 (PaddleOCR v3)
+-- tests/
|   +-- ocr_receiver_mjpeg.py     # 调试: 仅 OCR + 血量显示
|   +-- udp_sender.py             # 测试用 UDP 发送器
|   +-- legacy/                   # 旧脚本存档
+-- README.md
```

---

## 常见问题 / FAQ

**Q: 启动后显示 `等待 OBS 推流中...` 但没有反应？**

A: 确认 OBS 已开始推流，且 Output URL 中的 IP 和端口正确。如果 OBS 和本程序在同一台电脑上，使用 `udp://127.0.0.1:1234`。

**Q: OCR 识别结果不对，血量跳动很大？**

A: HPParser 内置了锁定 max 和跳变过滤机制。连续几帧会自动稳定。确保 OBS 输出分辨率是 256x64。

**Q: 手机扫码连接不上？**

A: 确保手机和本程序在同一局域网。`--ws` 参数的 IP 应该是手机能访问到的本机局域网 IP（不是 127.0.0.1）。

**Q: 波形太强/太弱？**

A: 调整 `config/ow_blood_pulse.yaml` 中的 `strength_min`、`strength_max` 和 `max_strength`。**`max_strength` 是安全硬上限，不要设置过高。**

**Q: 只有一台电脑，OBS 和本程序能同时运行吗？**

A: 可以。直接 `python main.py` 即可，OBS 推流到 `udp://127.0.0.1:1234`。

---

## 性能参考 / Performance

256x64 单帧 OCR (limit_side_len=64, cls=OFF):

| 配置 | 延迟 | 60 FPS |
|------|------|:------:|
| OpenVINO FP32 CPU | **~2ms** | Yes |
| RapidOCR (ONNX Runtime) | ~560ms | No |

关键优化：
- `limit_side_len=64`：默认 736 会将 64px 图放大到 736px，改为 64 后 OCR 延迟从 113ms -> 5ms
- `cls=OFF`：血量文本永远水平，跳过方向分类器节省 ~1.2ms
- CPU 比 GPU 快：本测试环境 i9-13900HX 远超 UHD 核显，强 iGPU/dGPU 下结论可能不同

---

## License

MIT
