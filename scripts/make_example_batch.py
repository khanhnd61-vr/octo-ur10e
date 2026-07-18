#!/usr/bin/env python3
"""
make_example_batch.py

Create the example_batch.pickle that OctoModelPt.load_pretrained() requires,
by adapting one saved from another finetune run of the same dataset.

Why this exists: a run dir copied without example_batch.pickle cannot be
loaded. The only window-dependent tensors are under observation/ and action/;
the task/ portion (all that inference actually uses, via create_tasks) does
not depend on window size. So a pickle from a window-1 run can be tiled along
the time dim to match a window-2 checkpoint.

Usage:
  python scripts/make_example_batch.py \
      --src /path/to/old_run/example_batch.pickle \
      --ckpt ckpts
"""
import argparse
import json
import pickle
from pathlib import Path

import torch


def tile_time_dim(batch, window):
    out = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = tile_time_dim(v, window)
        elif isinstance(v, torch.Tensor) and v.ndim >= 2 and v.shape[1] != window:
            reps = [1] * v.ndim
            reps[1] = window
            out[k] = v[:, :1].repeat(*reps)
        else:
            out[k] = v
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, help="example_batch.pickle from another run")
    p.add_argument("--ckpt", required=True, help="run dir to fix (holds config.json)")
    args = p.parse_args()

    ckpt = Path(args.ckpt)
    out_path = ckpt / "example_batch.pickle"
    if out_path.exists():
        raise SystemExit(f"{out_path} already exists; not overwriting.")

    with open(ckpt / "config.json") as f:
        window = json.load(f)["window_size"]

    with open(args.src, "rb") as f:
        eb = pickle.load(f)

    fixed = dict(eb)
    fixed["observation"] = tile_time_dim(eb["observation"], window)
    for k in ("action", "action_pad_mask"):
        if k in eb:
            fixed[k] = tile_time_dim({k: eb[k]}, window)[k]

    with open(out_path, "wb") as f:
        pickle.dump(fixed, f)

    mask = fixed["observation"]["timestep_pad_mask"]
    print(f"wrote {out_path}  (timestep_pad_mask {tuple(mask.shape)}, window={window})")


if __name__ == "__main__":
    main()
