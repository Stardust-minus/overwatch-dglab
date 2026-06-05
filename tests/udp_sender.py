#!/usr/bin/env python3
"""
UDP rawvideo 发送器 — 用来在本机测试 ocr_receiver.py
UDP rawvideo sender — for local testing of ocr_receiver.py

模拟 OBS 推流：读取图片或生成测试帧，通过 UDP 发送裸像素。
Simulates OBS streaming: reads images or generates test frames, sends raw pixels via UDP.

用法 / Usage:
    # 发送静态图片 (循环) / Send static image (loop)
    python udp_sender.py --image test.png

    # 发送随机生成的测试帧 / Send random test frames
    python udp_sender.py --random

    # 发送摄像头画面 / Send camera frames
    python udp_sender.py --camera 0
"""

import argparse
import socket
import time

import cv2
import numpy as np


def send_frames(frames, host="127.0.0.1", port=1234, fps=60):
    # 创建 UDP 套接字 // Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 计算帧间隔 // Calculate frame interval
    interval = 1.0 / fps

    print(f"Sending {len(frames)} frame(s) → {host}:{port} @ {fps} FPS")
    print(f"Frame size: {frames[0].nbytes} bytes ({frames[0].shape[1]}x{frames[0].shape[0]})")

    # 帧计数 // Frame counter
    count = 0
    t_start = time.perf_counter()

    try:
        while True:
            for frame in frames:
                # 确保尺寸正确 // Ensure correct dimensions
                if frame.shape[:2] != (64, 256):
                    frame = cv2.resize(frame, (256, 64))

                # 确保是 BGR 格式 // Ensure BGR format
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                # 序列化并发送 // Serialize and send
                data = frame.tobytes()
                sock.sendto(data, (host, port))
                count += 1

                # 每 60 帧打印一次统计 // Print stats every 60 frames
                if count % 60 == 0:
                    elapsed = time.perf_counter() - t_start
                    actual_fps = count / elapsed
                    print(f"\r  Sent {count} frames, {actual_fps:.1f} FPS", end="", flush=True)

                # 控制帧率 // Control frame rate
                t_send = time.perf_counter()
                wait = interval - (t_send - t_start - count * interval)
                if wait > 0:
                    time.sleep(wait)

    except KeyboardInterrupt:
        # 用户中断，打印总计 // User interrupted, print total
        print(f"\nStopped. Sent {count} frames total.")


def main():
    # 解析命令行参数 // Parse command-line arguments
    parser = argparse.ArgumentParser(description="UDP rawvideo sender (test)")
    parser.add_argument("--image", help="Static image to send in loop")
    parser.add_argument("--random", action="store_true", help="Send random test frames")
    parser.add_argument("--camera", type=int, help="Camera device index")
    parser.add_argument("--host", default="127.0.0.1", help="Receiver IP")
    parser.add_argument("--port", type=int, default=1234, help="Receiver port")
    parser.add_argument("--fps", type=int, default=60, help="Target FPS")
    args = parser.parse_args()

    # 待发送的帧列表 // List of frames to send
    frames = []

    if args.image:
        # 读取静态图片 // Read static image
        img = cv2.imread(args.image)
        if img is None:
            print(f"Error: Cannot read {args.image}")
            return
        img = cv2.resize(img, (256, 64))
        # 生成带帧号的序列 // Generate sequence with frame numbers
        for i in range(1):
            frame = img.copy()
            frames.append(frame)
        print(f"Loaded image: {args.image}")

    elif args.camera is not None:
        # 打开摄像头采集帧 // Open camera and capture frames
        cap = cv2.VideoCapture(args.camera)
        print(f"Opening camera {args.camera}...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (256, 64))
            frames.append(frame)
            if len(frames) >= 300:  # 缓冲 5 秒 // Buffer 5 seconds
                break
        cap.release()
        print(f"Captured {len(frames)} frames")

    elif args.random:
        # 生成随机测试帧（带文字）// Generate random test frames (with text)
        texts = [
            "Hello World Test",
            "Order #12345 $99.99",
            "Score: 8500 Level 42",
            "HP: 999 MP: 450",
            "Speed: 120 km/h",
        ]
        for text in texts:
            # 白色背景帧 // White background frame
            frame = np.ones((64, 256, 3), dtype=np.uint8) * 255
            cv2.putText(frame, text, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            frames.append(frame)
        print(f"Generated {len(frames)} test frames")

    else:
        # 默认：生成一组测试帧 // Default: generate a set of test frames
        for i in range(5):
            # 白色背景 + 帧编号 // White background with frame number
            frame = np.ones((64, 256, 3), dtype=np.uint8) * 255
            cv2.putText(frame, f"Frame {i:03d}", (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            frames.append(frame)
        print(f"Generated {len(frames)} default test frames")

    # 开始发送帧 // Start sending frames
    send_frames(frames, host=args.host, port=args.port, fps=args.fps)


if __name__ == "__main__":
    main()
