#!/usr/bin/env python3
"""
open_loop_eval.py

Open-loop evaluation of a finetuned Octo checkpoint against the dataset it was
trained on, in the style of Isaac-GR00T's `scripts/eval_policy.py`.

"Open loop" = we never feed the policy's own actions back into the robot. We walk
the *recorded* episode, and every `action_horizon` steps we show the policy the
recorded observation and ask it to predict the next `action_horizon` actions. Those
predictions are stitched together and compared against the recorded ground-truth
actions for the whole episode. This measures how well the policy reproduces the
demonstrations without needing a simulator or the real robot.

Outputs, per episode:
  * per-dimension MSE / MAE between predicted and ground-truth actions
  * a matplotlib figure: one subplot per action dimension, ground truth (solid)
    vs prediction (dashed), x-axis = timestep

Action layout (this dataset, bridge/OXE EEF_POS convention):
  [dx, dy, dz, droll, dpitch, dyaw, gripper]   gripper absolute binary (1=open)

Usage:
  python open_loop_eval.py \
      --checkpoint checkpoints/octo_finetune/ur10e_pick_cup_20260714_003435 \
      --data-dir rlds --dataset-name ur10e_pick_cup \
      --episodes 3 --out eval_out
"""
import argparse
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# The policy runs on the GPU via torch; TF is only used to read the RLDS shards.
os.environ.setdefault("CUDA_VISIBLE_DEVICES_TF", "")

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import tensorflow_datasets as tfds
    import torch
except ImportError as e:
    sys.exit(f"Missing dep: {e}")

# Keep TF off the GPU; the policy needs the VRAM.
tf.config.set_visible_devices([], "GPU")
try:
    tfds.core.utils.gcs_utils._is_gcs_disabled = True
except Exception:
    pass

from octo.model.octo_model_pt import OctoModelPt

ACTION_DIM_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


def get_action_stats(model):
    """Return the {'mean','std','mask',...} action stats, tolerating both the flat
    layout saved by finetuning and the {dataset_name: {...}} layout of the JAX
    pretrained checkpoints."""
    stats = model.dataset_statistics
    if "action" in stats:
        return stats["action"]
    if len(stats) == 1:
        return next(iter(stats.values()))["action"]
    raise ValueError(
        f"Ambiguous dataset_statistics with keys {list(stats)}; expected a single dataset."
    )


def load_episode(steps):
    """RLDS steps -> numpy arrays for one episode."""
    primary, wrist, actions = [], [], []
    language = None
    for s in steps:
        obs = s["observation"]
        primary.append(obs["image"].numpy())
        wrist.append(obs["wrist_image"].numpy())
        actions.append(s["action"].numpy())
        if language is None:
            language = s["language_instruction"].numpy().decode()
    return (
        np.asarray(primary),   # (T, 256, 256, 3) uint8
        np.asarray(wrist),     # (T, 128, 128, 3) uint8
        np.asarray(actions),   # (T, 7) float32
        language,
    )


def build_observation(primary, wrist, t, window_size, device):
    """Observation window ending at timestep t, shaped the way OctoModulePt wants:
    images (1, window, 3, H, W) uint8, plus a pad mask marking slots that fall off
    the start of the episode."""
    idxs = [max(0, t - (window_size - 1) + i) for i in range(window_size)]
    valid = [(t - (window_size - 1) + i) >= 0 for i in range(window_size)]

    def stack(frames):
        # (window, H, W, 3) -> (1, window, 3, H, W)
        arr = np.stack([frames[i] for i in idxs]).transpose(0, 3, 1, 2)
        return torch.from_numpy(arr).unsqueeze(0).to(device)

    return {
        "image_primary": stack(primary),
        "image_wrist": stack(wrist),
        "timestep_pad_mask": torch.tensor([valid], dtype=torch.bool, device=device),
    }


@torch.no_grad()
def rollout_episode(model, primary, wrist, gt_actions, language, args, device):
    """Open-loop: predict an action chunk every `action_horizon` recorded steps."""
    T = len(gt_actions)
    window_size = model.example_batch["observation"]["timestep_pad_mask"].shape[1]
    action_stats = get_action_stats(model)
    task = model.create_tasks(texts=[language], device=device)
    generator = torch.Generator(device).manual_seed(args.seed)

    pred = np.full_like(gt_actions, np.nan, dtype=np.float32)
    for t in range(0, T, args.action_horizon):
        obs = build_observation(primary, wrist, t, window_size, device)
        chunk = model.sample_actions(
            obs,
            task,
            unnormalization_statistics=action_stats,
            generator=generator,
        )
        chunk = np.asarray(chunk.squeeze().cpu())  # (action_horizon, action_dim)
        if chunk.ndim == 1:
            chunk = chunk[None]
        # place the chunk over the steps it is meant to cover, clipped at the end
        n = min(args.action_horizon, T - t, len(chunk))
        pred[t : t + n] = chunk[:n]

    return pred


def plot_episode(gt, pred, ep_name, language, mse, out_path):
    n_dim = gt.shape[1]
    fig, axes = plt.subplots(n_dim, 1, figsize=(12, 2.0 * n_dim), sharex=True)
    steps = np.arange(len(gt))
    for d in range(n_dim):
        ax = axes[d]
        ax.plot(steps, gt[:, d], "-", lw=1.8, color="#1f77b4", label="ground truth")
        ax.plot(steps, pred[:, d], "--", lw=1.5, color="#d62728", label="predicted")
        name = ACTION_DIM_NAMES[d] if d < len(ACTION_DIM_NAMES) else f"dim{d}"
        ax.set_ylabel(f"{name}\nMSE={mse[d]:.2e}", fontsize=8)
        ax.grid(alpha=0.3)
        if d == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("timestep")
    fig.suptitle(
        f"{ep_name} — open-loop  |  task: {language!r}  |  overall MSE={mse.mean():.3e}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint", required=True,
                   help="Finetune run dir (holds config.json + step subdirs)")
    p.add_argument("--step", type=int, default=None,
                   help="Checkpoint step to load (default: latest)")
    p.add_argument("--data-dir", default="rlds", help="TFDS data_dir")
    p.add_argument("--dataset-name", default="ur10e_pick_cup")
    p.add_argument("--split", default="train")
    p.add_argument("--episodes", type=int, default=3,
                   help="How many episodes to evaluate (default: 3)")
    p.add_argument("--action-horizon", type=int, default=4,
                   help="Actions predicted per policy query (default: 4, matches training)")
    p.add_argument("--out", default="eval_out", help="Directory for plots")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    print(f"Loading checkpoint {args.checkpoint} (step={args.step or 'latest'})")
    model = OctoModelPt.load_pretrained(args.checkpoint, step=args.step)["octo_model"]
    model = model.to(device).eval()

    stats = get_action_stats(model)
    print("action mean:", np.asarray(stats["mean"]).round(5))
    print("action std :", np.asarray(stats["std"]).round(5))

    builder = tfds.builder_from_directory(
        builder_dir=os.path.join(args.data_dir, args.dataset_name, "1.0.0")
    )
    ds = builder.as_dataset(split=args.split)

    all_mse, all_mae = [], []
    for i, ep in enumerate(ds.take(args.episodes)):
        primary, wrist, gt, language = load_episode(ep["steps"])
        ep_name = f"episode_{i:03d}"
        print(f"\n=== {ep_name}: T={len(gt)}  task={language!r}")

        pred = rollout_episode(model, primary, wrist, gt, language, args, device)

        mse = np.mean((pred - gt) ** 2, axis=0)
        mae = np.mean(np.abs(pred - gt), axis=0)
        all_mse.append(mse)
        all_mae.append(mae)

        for d, name in enumerate(ACTION_DIM_NAMES[: gt.shape[1]]):
            print(
                f"  {name:8s} MSE={mse[d]:.3e}  MAE={mae[d]:.3e}  "
                f"| gt std={gt[:, d].std():.3e}  pred std={pred[:, d].std():.3e}"
            )
        print(f"  overall  MSE={mse.mean():.3e}  MAE={mae.mean():.3e}")

        out_path = os.path.join(args.out, f"{ep_name}.png")
        plot_episode(gt, pred, ep_name, language, mse, out_path)
        print(f"  plot -> {out_path}")

    mse = np.mean(all_mse, axis=0)
    mae = np.mean(all_mae, axis=0)
    print(f"\n===== mean over {len(all_mse)} episodes =====")
    for d, name in enumerate(ACTION_DIM_NAMES[: len(mse)]):
        print(f"  {name:8s} MSE={mse[d]:.3e}  MAE={mae[d]:.3e}")
    print(f"  overall  MSE={mse.mean():.3e}  MAE={mae.mean():.3e}")


if __name__ == "__main__":
    main()
