#!/usr/bin/env bash
# Launch octo-small-1.5 finetuning on the UR10e pick-cup RLDS dataset.
# Designed to run inside a detached screen (see how it's started in the notes).
set -euo pipefail

REPO=/home/khanh/work/octo-pytorch
cd "$REPO"
source .venv/bin/activate

export TF_CPP_MIN_LOG_LEVEL=2          # quiet TF C++ logs
# TF is CPU-only for data loading; finetune_pt.py already hides the GPU from TF.

# wandb online: authenticate via WANDB_API_KEY from the environment.
# (The account's new-style `wandb_v1_` key is 86 chars; `wandb login` / netrc
#  reject it on this wandb build, but the server accepts it via env var.)
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "ERROR: WANDB_API_KEY not set in environment; cannot log to wandb online." >&2
  echo "       export WANDB_API_KEY=... before launching, or set WANDB_MODE=offline." >&2
  exit 1
fi

# Overridable so the same runner drives both the no-proprio and proprio variants.
CONFIG="${CONFIG:-scripts/configs/finetune_ur10e.py}"
NAME="${NAME:-ur10e_pick_cup}"
SAVE_DIR="${SAVE_DIR:-/mnt/data/ur10e-robotiq/octo-small-finetune/checkpoints}"
LOG="${LOG:-/mnt/data/ur10e-robotiq/octo-small-finetune/finetune_ur10e.log}"
mkdir -p "$SAVE_DIR" "$(dirname "$LOG")"

# 1 epoch = 20596 transitions / batch 16 = ~1287 steps.
# 60000 steps = ~46 epochs (the baseline run converged by ~40k).
# The cosine LR anneals to zero exactly at NUM_STEPS, so this also sets the schedule.
NUM_STEPS="${NUM_STEPS:-30000}"

echo "=== launch $(date)  config=$CONFIG  name=$NAME  num_steps=$NUM_STEPS ===" | tee -a "$LOG"

torchrun --nproc_per_node 1 scripts/finetune_pt.py \
  --name "$NAME" \
  --config "${CONFIG}:full,language_conditioned" \
  --config.pretrained_path=hf://rail-berkeley/octo-small-1.5 \
  --config.batch_size=16 \
  --config.num_steps="$NUM_STEPS" \
  --config.save_interval=2500 \
  --config.log_interval=100 \
  --config.save_dir="$SAVE_DIR" \
  2>&1 | tee -a "$LOG"

echo "=== finished $(date) ===" | tee -a "$LOG"
