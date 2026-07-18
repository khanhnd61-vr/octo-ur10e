#!/usr/bin/env python3
"""
octo_policy_server.py

TCP inference server for a finetuned Octo checkpoint. Runs on the host in the
octo-ur10e venv (torch + GPU); the ROS 2 side (vr_ur_teleop container) sends
raw camera frames and gets back one un-normalized action chunk.

All model-specific preprocessing lives here so the ROS client stays dumb:
  * frames are squash-resized to 256x256 (primary) / 128x128 (wrist) with
    cv2.INTER_AREA -- byte-for-byte the same as convert_ur_to_rlds.to_square,
    which is what the training data went through
  * observation window (1, window, 3, H, W) uint8 + timestep_pad_mask
  * sample_actions with the checkpoint's dataset_statistics["action"]
    (gripper dim unmasked -> stays ~0/1)

Protocol (one TCP connection, many requests):
  request  = 4-byte big-endian length + pickle of
             {"primary": [HxWx3 uint8]*window, "wrist": [...]*window,
              "pad": [bool]*window, "text": str (optional)}
  response = same framing, pickle of
             {"actions": (action_horizon, 7) float32 ndarray}  or  {"error": str}

Action layout (bridge/OXE EEF_POS):
  [dx, dy, dz, droll, dpitch, dyaw, gripper]
  deltas in m/rad per policy step (~stride/camera_hz seconds), world frame;
  gripper absolute (1=open, 0=closed).

Usage:
  .venv/bin/python scripts/octo_policy_server.py --checkpoint ckpts
"""
import argparse
import os
import pickle
import socket
import struct
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

try:
    import cv2
    import tensorflow as tf
    import torch
except ImportError as e:
    sys.exit(f"Missing dep: {e}. Run inside the octo-ur10e venv.")

# TF is only pulled in transitively; keep it off the GPU.
tf.config.set_visible_devices([], "GPU")

from octo.model.octo_model_pt import OctoModelPt


def to_square(img, size, mode):
    """Same as convert_ur_to_rlds.to_square. HxWx3 uint8 RGB -> size x size x 3."""
    if mode == "crop":
        h, w = img.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        img = img[y0:y0 + s, x0:x0 + s]
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def get_action_stats(model):
    stats = model.dataset_statistics
    if "action" in stats:
        return stats["action"]
    if len(stats) == 1:
        return next(iter(stats.values()))["action"]
    raise ValueError(f"Ambiguous dataset_statistics keys: {list(stats)}")


def recv_msg(conn):
    hdr = conn.recv(4, socket.MSG_WAITALL)
    if len(hdr) < 4:
        return None
    (n,) = struct.unpack(">I", hdr)
    buf = conn.recv(n, socket.MSG_WAITALL)
    if len(buf) < n:
        return None
    return pickle.loads(buf)


def send_msg(conn, obj):
    buf = pickle.dumps(obj)
    conn.sendall(struct.pack(">I", len(buf)) + buf)


class Policy:
    def __init__(self, args):
        device = args.device if torch.cuda.is_available() else "cpu"
        print(f"Loading {args.checkpoint} (step={args.step or 'latest'}) on {device} ...")
        self.model = OctoModelPt.load_pretrained(args.checkpoint, step=args.step)["octo_model"]
        self.model = self.model.to(device).eval()
        self.device = device
        self.stats = get_action_stats(self.model)
        self.window = self.model.example_batch["observation"]["timestep_pad_mask"].shape[1]
        self.img_mode = args.img_mode
        self.primary_size = args.primary_size
        self.wrist_size = args.wrist_size
        self.generator = torch.Generator(device).manual_seed(args.seed)
        self._task_text = None
        self._task = None
        self.set_task(args.task)

        # First sample_actions call compiles/warms up (~9 s); do it before serving
        # so the robot loop never blocks on it.
        t0 = time.time()
        self.predict(
            [np.zeros((480, 640, 3), np.uint8)] * self.window,
            [np.zeros((480, 640, 3), np.uint8)] * self.window,
            [True] * self.window,
        )
        print(f"Warmed up in {time.time() - t0:.1f} s. window={self.window}, task={self._task_text!r}")

    def set_task(self, text):
        if text != self._task_text:
            self._task = self.model.create_tasks(texts=[text], device=self.device)
            self._task_text = text
            print(f"Task set: {text!r}")

    def _stack(self, frames, size):
        imgs = [to_square(f, size, self.img_mode) for f in frames]
        arr = np.stack(imgs).transpose(0, 3, 1, 2)  # (window, 3, H, W)
        return torch.from_numpy(arr).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, primary, wrist, pad):
        assert len(primary) == len(wrist) == len(pad) == self.window, (
            f"need {self.window} frames per camera, got "
            f"{len(primary)}/{len(wrist)}/{len(pad)}"
        )
        obs = {
            "image_primary": self._stack(primary, self.primary_size),
            "image_wrist": self._stack(wrist, self.wrist_size),
            "timestep_pad_mask": torch.tensor([pad], dtype=torch.bool, device=self.device),
        }
        actions = self.model.sample_actions(
            obs, self._task,
            unnormalization_statistics=self.stats,
            generator=self.generator,
        )
        return np.asarray(actions.squeeze(0).cpu(), dtype=np.float32)  # (horizon, 7)


def serve(policy, host, port):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    print(f"Octo policy server listening on {host}:{port}")
    while True:
        conn, addr = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"Client connected: {addr}")
        send_msg(conn, {"window": policy.window, "task": policy._task_text})
        try:
            n = 0
            while True:
                req = recv_msg(conn)
                if req is None:
                    break
                try:
                    if req.get("text"):
                        policy.set_task(req["text"])
                    t0 = time.time()
                    actions = policy.predict(req["primary"], req["wrist"], req["pad"])
                    send_msg(conn, {"actions": actions})
                    n += 1
                    if n % 25 == 1:
                        print(f"query {n}: {(time.time() - t0) * 1000:.0f} ms, "
                              f"chunk[0]={actions[0].round(4)}")
                except Exception as e:  # noqa: BLE001
                    print(f"request failed: {e}")
                    send_msg(conn, {"error": str(e)})
        finally:
            conn.close()
            print(f"Client disconnected: {addr} ({n} queries served)")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="ckpts",
                   help="Run dir with config.json, dataset_statistics.json, "
                        "example_batch.pickle and step subdirs (default: ckpts)")
    p.add_argument("--step", type=int, default=None, help="Checkpoint step (default: latest)")
    p.add_argument("--task", default="pick up the cup",
                   help="Language instruction; must match training data")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8790)
    p.add_argument("--img-mode", choices=["resize", "crop"], default="resize",
                   help="Must match the --img-mode used in convert_ur_to_rlds.py (default: resize)")
    p.add_argument("--primary-size", type=int, default=256)
    p.add_argument("--wrist-size", type=int, default=128)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    policy = Policy(args)
    serve(policy, args.host, args.port)


if __name__ == "__main__":
    main()
