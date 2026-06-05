#!/usr/bin/env python3
"""
守望先锋掉血 & 死亡 → 郊狼波形控制

Overwatch HP loss & death → DG-LAB waveform control

完整流水线:
  OBS (MJPEG/UDP) → OCR → HP 解析 → 血量事件 → 郊狼波形输出

Pipeline:
  OBS (MJPEG/UDP) → OCR → HP parse → HP event → DG-LAB waveform output

用法 / Usage:
  python main.py                                 # 默认配置 (config/ow_blood_pulse.yaml)
  python main.py -c my_config.yaml               # 指定配置文件
  python main.py --ws ws://IP:5678               # 命令行覆盖

  启动后会显示郊狼二维码，用 DG-LAB APP 扫码绑定。
  然后 OBS 开始推流即可。
"""

import argparse
import asyncio
import os
import sys
import socket
import time
from enum import Enum
from typing import Optional

# 将 lib/ 加入模块搜索路径 // Add lib/ to module search path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))

import cv2
import numpy as np
import yaml

# ── 帧参数 / Frame params ────────────────────────────────────
W, H = 256, 64
JPEG_START = b'\xff\xd8'
JPEG_END = b'\xff\xd9'


# ── 血量事件 / HP events ────────────────────────────────────
class HPEvent(Enum):
    FULL = "full"            # 满血 / Full HP
    DAMAGED = "damaged"      # 掉血 / HP loss
    HEALED = "healed"        # 回血 / HP restore
    DEATH = "death"          # 死亡 / Death
    UNCHANGED = "unchanged"  # 无变化 / No change


# ── 安全限制 / Safety limit ─────────────────────────────────
# set_strength 的绝对硬上限，代码中不可配置超过此值
# Absolute hard cap for set_strength, cannot be exceeded in code
HARD_STRENGTH_CAP = 30


def load_config(path: str) -> dict:
    """加载 YAML 配置文件，返回配置字典。"""
    """Load YAML config file and return config dict."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # 解析波形数据: list → tuple // Parse wave data: list → tuple
    for name, wave in cfg.get("waves", {}).items():
        data = wave.get("data", [])
        wave["data"] = [((tuple(d[0]), tuple(d[1]))) for d in data]
    return cfg


def build_wave(name: str, cfg: dict) -> tuple:
    """从配置构建波形: (wave_data, repeat)。"""
    """Build waveform from config: (wave_data, repeat)."""
    waves = cfg.get("waves", {})
    if name not in waves:
        available = ", ".join(waves.keys())
        raise ValueError(f"波形 '{name}' 未定义, 可用: {available}")
    w = waves[name]
    return w["data"], w.get("repeat", 1)


class BloodPulseController:
    """
    血量事件 → DG-LAB 波形控制器。
    HP event → DG-LAB waveform controller.

    队列控制:
    Queue control:
      - 每次新事件先 clear_pulses 清空设备队列，再发新波形
      - Clear device queue via clear_pulses on each new event, then send new waveform
      - 队列深度跟踪，超过上限自动清空
      - Track queue depth, auto-clear when exceeding limit
    """

    MAX_QUEUE_SECONDS = 2.0

    def __init__(self, dg_client, channel="A", cfg: Optional[dict] = None):
        from pydglab_ws import Channel

        cfg = cfg or {}
        self.dg = dg_client
        self.channel = Channel.A if channel == "A" else Channel.B

        # 安全: max_strength 不超过硬上限 // Safety: max_strength cannot exceed hard cap
        self.max_strength = min(cfg.get("max_strength", HARD_STRENGTH_CAP), HARD_STRENGTH_CAP)

        # 事件参数 // Event parameters
        self.damage_threshold = cfg.get("damage_threshold", 1)
        self.low_hp_ratio = cfg.get("low_hp_ratio", 0.3)
        self.damage_cooldown = cfg.get("damage_cooldown", 0.8)
        self.respawn_seconds = cfg.get("respawn_seconds", 12.0)

        # 强度映射 // Strength mapping
        self.strength_min = min(cfg.get("strength_min", 8), self.max_strength)
        self.strength_max = min(cfg.get("strength_max", 30), self.max_strength)

        # 仅死亡模式 // Death-only mode
        self.death_only = cfg.get("death_only", False)

        # 波形 // Waveforms
        self.damage_wave_name = cfg.get("damage_wave", "pinch")
        self.low_hp_wave_name = cfg.get("low_hp_wave", "heartbeat")
        self.death_wave_name = cfg.get("death_wave", "death_shock")
        self._cfg = cfg

        # 状态 // State
        self.prev_hp: Optional[int] = None
        self.max_hp: Optional[int] = None
        self.is_dead = False
        self.death_time: float = 0.0
        self.last_damage_time: float = 0.0
        self._pulse_task: Optional[asyncio.Task] = None

        # 队列跟踪 // Queue tracking
        self._queue_entries: int = 0
        self._max_queue_entries: int = int(self.MAX_QUEUE_SECONDS / 0.1)

    def _clamp(self, value: int) -> int:
        """强度安全钳位。"""
        """Clamp strength to safe range."""
        return max(0, min(value, self.max_strength))

    def _calc_damage_strength(self, ratio: float) -> int:
        """按血量比例计算掉血强度: ratio=1.0→min, ratio=0.0→max"""
        """Calc damage strength by HP ratio: ratio=1.0→min, ratio=0.0→max"""
        raw = self.strength_min + (self.strength_max - self.strength_min) * (1 - ratio)
        return self._clamp(int(raw))

    def detect_event(self, current: int, max_hp: int) -> HPEvent:
        """根据 HP 变化检测事件。"""
        """Detect event based on HP change."""
        if self.max_hp is None:
            self.max_hp = max_hp
            self.prev_hp = current
            return HPEvent.UNCHANGED

        self.max_hp = max_hp

        # 死亡 / Death
        if current == 0 and not self.is_dead:
            self.is_dead = True
            self.death_time = time.monotonic()
            self.prev_hp = current
            return HPEvent.DEATH

        # 已死亡状态 / Already dead state
        if self.is_dead:
            elapsed = time.monotonic() - self.death_time
            if elapsed < self.respawn_seconds:
                return HPEvent.UNCHANGED
            if current > 0:
                self.is_dead = False
                self.prev_hp = current
                return HPEvent.HEALED
            return HPEvent.UNCHANGED

        # 无变化 / No change
        if self.prev_hp is not None and current == self.prev_hp:
            return HPEvent.UNCHANGED

        # 掉血 (冷却期内只更新 prev_hp) / HP loss (only update prev_hp during cooldown)
        if self.prev_hp is not None and current < self.prev_hp:
            delta = self.prev_hp - current
            if delta >= self.damage_threshold:
                self.prev_hp = current
                if time.monotonic() - self.last_damage_time < self.damage_cooldown:
                    return HPEvent.UNCHANGED
                self.last_damage_time = time.monotonic()
                return HPEvent.DAMAGED

        # 回血 / HP restore
        if self.prev_hp is not None and current > self.prev_hp:
            self.prev_hp = current
            return HPEvent.HEALED

        self.prev_hp = current
        return HPEvent.UNCHANGED

    async def handle_event(self, event: HPEvent, current: int, max_hp: int):
        """根据事件执行对应操作。"""
        """Execute action based on event."""
        from pydglab_ws import StrengthOperationType

        if event == HPEvent.DEATH:
            strength = self._clamp(self.max_strength)
            wave_data, repeat = build_wave(self.death_wave_name, self._cfg)
            wave_ms = len(wave_data) * repeat * 100
            print(f"\n  [DEATH] HP=0/{max_hp} -> 强度={strength}, {self.death_wave_name}({wave_ms}ms), {self.respawn_seconds:.0f}s内禁用")
            await self._stop_pulse()
            await self._clear_queue()
            await self.dg.set_strength(self.channel, StrengthOperationType.SET_TO, strength)
            self._pulse_task = asyncio.create_task(self._send_once(wave_data, repeat))

        elif event == HPEvent.DAMAGED:
            # 仅死亡模式下跳过掉血波形 / Skip damage waveform in death-only mode
            if self.death_only:
                return
            ratio = current / max_hp if max_hp > 0 else 1.0
            strength = self._calc_damage_strength(ratio)
            is_low = ratio < self.low_hp_ratio
            wave_name = self.low_hp_wave_name if is_low else self.damage_wave_name
            wave_data, repeat = build_wave(wave_name, self._cfg)
            print(f"\n  [DAMAGE] HP={current}/{max_hp} ({ratio:.0%}) -> 强度={strength}, {wave_name}")
            await self._stop_pulse()
            await self._clear_queue()
            await self.dg.set_strength(self.channel, StrengthOperationType.SET_TO, strength)
            self._pulse_task = asyncio.create_task(self._send_once(wave_data, repeat))

        elif event == HPEvent.HEALED:
            # 仅死亡模式下跳过回血处理 / Skip heal handling in death-only mode
            if self.death_only:
                return
            print(f"\n  [HEAL] HP={current}/{max_hp} -> 停止波形")
            await self._stop_pulse()
            await self._clear_queue()

        elif event == HPEvent.FULL:
            # 仅死亡模式下跳过满血处理 / Skip full-HP handling in death-only mode
            if self.death_only:
                return
            print(f"\n  [FULL] HP={current}/{max_hp} -> 清空波形, 强度归零")
            await self._stop_pulse()
            await self._clear_queue()
            try:
                await self.dg.set_strength(self.channel, StrengthOperationType.SET_TO, 0)
            except Exception:
                pass

    async def _clear_queue(self):
        """清空设备波形队列并重置计数。"""
        """Clear device waveform queue and reset counter."""
        self._queue_entries = 0
        try:
            await self.dg.clear_pulses(self.channel)
        except Exception:
            pass

    async def _add_with_tracking(self, entries: list):
        """添加波形数据并跟踪队列深度。"""
        """Add waveform data and track queue depth."""
        n = len(entries)
        if self._queue_entries + n > self._max_queue_entries:
            await self._clear_queue()
        await self.dg.add_pulses(self.channel, *entries)
        self._queue_entries += n

    async def _send_once(self, wave, repeat: int = 1):
        """发送波形 N 次然后停止。"""
        """Send waveform N times then stop."""
        try:
            entries = list(wave * repeat)
            await self._add_with_tracking(entries)
        except Exception as e:
            print(f"  [波形发送失败] {e}")

    async def _stop_pulse(self):
        """停止正在发送的波形。"""
        """Stop the currently sending waveform."""
        if self._pulse_task and not self._pulse_task.done():
            self._pulse_task.cancel()
            try:
                await self._pulse_task
            except asyncio.CancelledError:
                pass
        self._pulse_task = None


# ── MJPEG 接收器 / MJPEG receiver ──────────────────────────
class MJPEGReceiver:
    def __init__(self, port=1234):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
        self.sock.bind(("0.0.0.0", port))
        self.sock.settimeout(5.0)
        self.buffer = b''
        self.port = port

    def recv_frame(self):
        while True:
            start = self.buffer.find(JPEG_START)
            if start != -1:
                end = self.buffer.find(JPEG_END, start)
                if end != -1:
                    jpeg_data = self.buffer[start:end + 2]
                    self.buffer = self.buffer[end + 2:]
                    frame = cv2.imdecode(np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        return frame
                    continue
            try:
                data, _ = self.sock.recvfrom(65535)
                self.buffer += data
                if len(self.buffer) > 10 * 1024 * 1024:
                    last_start = self.buffer.rfind(JPEG_START)
                    if last_start > 0:
                        self.buffer = self.buffer[last_start:]
                    else:
                        self.buffer = self.buffer[-1 * 1024 * 1024:]
            except socket.timeout:
                return None

    def close(self):
        self.sock.close()


# ── 主程序 / Main ──────────────────────────────────────────
async def main():
    default_cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "ow_blood_pulse.yaml")

    parser = argparse.ArgumentParser(description="守望先锋掉血&死亡 → 郊狼波形控制")
    parser.add_argument("-c", "--config", default=default_cfg, help="配置文件路径")
    parser.add_argument("--ws", default=None, help="郊狼 WS 地址 (覆盖配置文件)")
    parser.add_argument("--port", type=int, default=None, help="OBS UDP 端口 (覆盖配置文件)")
    parser.add_argument("--channel", default=None, choices=["A", "B"], help="输出通道 (覆盖配置文件)")
    parser.add_argument("--max-strength", type=int, default=None, help=f"强度上限 (覆盖配置文件, 硬上限 {HARD_STRENGTH_CAP})")
    parser.add_argument("--death-only", action="store_true", default=None, help="仅启用死亡波形, 忽略掉血/回血/满血")
    args = parser.parse_args()

    # 加载配置 // Load config
    cfg = load_config(args.config)

    # 命令行覆盖 // CLI overrides
    if args.ws is not None:
        cfg["ws_address"] = args.ws
    if args.port is not None:
        cfg["udp_port"] = args.port
    if args.channel is not None:
        cfg["channel"] = args.channel
    if args.max_strength is not None:
        cfg["max_strength"] = args.max_strength
    if args.death_only:
        cfg["death_only"] = True

    # ── 1. 初始化 DG-LAB / Init DG-LAB ─────────────────────
    import qrcode
    import io
    from pydglab_ws import DGLabWSServer

    ws_url = cfg.get("ws_address", "ws://127.0.0.1:5678")
    ws_port = cfg.get("ws_port", 5678)

    print("=" * 60)
    print("  守望先锋掉血 & 死亡 -> 郊狼波形控制")
    print("=" * 60)

    if ws_url.startswith("ws://"):
        parts = ws_url[5:].split(":")
        ws_host = parts[0]
        ws_port = int(parts[1]) if len(parts) > 1 else ws_port
    else:
        ws_host = "0.0.0.0"

    print(f"\n  启动郊狼 WebSocket 服务 (端口 {ws_port})...")
    print(f"  配置文件: {args.config}")
    async with DGLabWSServer("0.0.0.0", ws_port, 60) as server:
        client = server.new_local_client()

        # 生成二维码 // Generate QR code
        qr_url = client.get_qrcode(f"ws://{ws_host}:{ws_port}")
        print("\n  请用 DG-LAB APP 扫描二维码绑定郊狼:")
        qr = qrcode.QRCode()
        qr.add_data(qr_url)
        f = io.StringIO()
        qr.print_ascii(out=f)
        f.seek(0)
        print(f.read())
        print(f"  或手动输入: {qr_url}\n")

        # 等待绑定 // Wait for binding
        print("  等待 APP 扫码绑定...")
        await client.bind()
        print(f"  [OK] 绑定成功! targetId={client.target_id}\n")

        # ── 2. 初始化 OCR / Init OCR ────────────────────────
        from rapidocr_openvino import create_rapidocr_with_openvino
        from rapidocr_onnxruntime.ch_ppocr_v3_det.utils import create_operators
        from hp_parser import HPParser

        ocr = create_rapidocr_with_openvino(device="CPU")
        ocr.text_detector.preprocess_op = create_operators({
            'DetResizeForTest': {'limit_side_len': 64, 'limit_type': 'min'},
            'NormalizeImage': {'std': [0.229, 0.224, 0.225], 'mean': [0.485, 0.456, 0.406],
                               'scale': '1./255.', 'order': 'hwc'},
            'ToCHWImage': None,
            'KeepKeys': {'keep_keys': ['image', 'shape']},
        })
        # 血量文本永远是水平的，跳过方向分类器省 ~1.2ms
        # HP text is always horizontal, skip angle classifier to save ~1.2ms
        ocr.use_angle_cls = False
        hp_parser = HPParser()
        controller = BloodPulseController(
            client,
            channel=cfg.get("channel", "A"),
            cfg=cfg,
        )
        print("  [OCR] OpenVINO FP32 CPU, limit_side_len=64, cls=OFF")
        print(f"  [DG-LAB] 通道={cfg.get('channel', 'A')}, 强度上限={controller.max_strength}")
        mode_str = "仅死亡" if controller.death_only else "全部"
        print(f"  [模式] {mode_str} | 波形: 掉血={controller.damage_wave_name}, 低血={controller.low_hp_wave_name}, 死亡={controller.death_wave_name}")

        # ── 3. 启动接收 + OCR + 控制循环 / Start recv + OCR + control loop ──
        udp_port = cfg.get("udp_port", 1234)
        receiver = MJPEGReceiver(port=udp_port)
        print(f"\n  监听 UDP :{udp_port}, 等待 OBS 推流... (Ctrl+C 退出)\n")
        print("-" * 60)

        frame_count = 0
        ocr_times = []
        last_display_event = ""

        try:
            # DG-LAB 心跳/断线检测（后台任务） // DG-LAB heartbeat/disconnect detection (background task)
            async def dg_monitor():
                from pydglab_ws import StrengthData, RetCode
                async for data in client.data_generator():
                    if isinstance(data, StrengthData):
                        pass
                    elif data == RetCode.CLIENT_DISCONNECTED:
                        print("\n  [WARN] APP 断开! 等待重连...")
                        await client.rebind()
                        print("  [OK] 重连成功")

            monitor_task = asyncio.create_task(dg_monitor())

            # 主 OCR 循环 / Main OCR loop
            # recv_frame 是同步 UDP 读取，直接在事件循环中调用
            # recv_frame is a sync UDP read, called directly in the event loop
            # (非阻塞: timeout=5s, 但通常 <1ms 就返回帧)
            # (non-blocking: timeout=5s, but usually returns a frame in <1ms)
            # OCR 也直接调用: ~3ms 不会显著阻塞事件循环
            # OCR is also called directly: ~3ms won't significantly block the event loop
            # 避免了 run_in_executor 的线程切换开销 (~0.4ms)
            # Avoids run_in_executor thread switching overhead (~0.4ms)
            while True:
                frame = receiver.recv_frame()
                if frame is None:
                    continue

                t0 = time.perf_counter()
                result, _ = ocr(frame)
                ocr_ms = (time.perf_counter() - t0) * 1000
                ocr_times.append(ocr_ms)
                frame_count += 1

                # HP 解析 / HP parsing
                ocr_texts = [r[1] for r in result] if result else []
                hp = hp_parser.parse(ocr_texts)

                if hp.valid:
                    event = controller.detect_event(hp.current, hp.max_hp)
                    last_display_event = f"HP={hp.current}/{hp.max_hp} [{hp.confidence}]"
                    if event != HPEvent.UNCHANGED:
                        await controller.handle_event(event, hp.current, hp.max_hp)
                        last_display_event += f" {event.value}"
                else:
                    last_display_event = f"HP=--- raw={hp.raw!r}"

                avg_ocr = np.mean(ocr_times[-60:]) if ocr_times else 0
                print(f"\r  Frame {frame_count}: ocr={ocr_ms:.1f}ms(avg={avg_ocr:.1f}ms) {last_display_event}",
                      end="", flush=True)

        except KeyboardInterrupt:
            pass
        finally:
            monitor_task.cancel()
            receiver.close()
            try:
                from pydglab_ws import StrengthOperationType
                await controller._stop_pulse()
                await client.clear_pulses(controller.channel)
                await client.set_strength(controller.channel, StrengthOperationType.SET_TO, controller._clamp(0))
                print("\n\n  [OK] 安全关闭: 波形清空, 强度归零")
            except Exception:
                print("\n\n  [WARN] 关闭时清空失败，请手动检查设备")


if __name__ == "__main__":
    asyncio.run(main())
