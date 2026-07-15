# Open-Loop Evaluation on Another Machine

How to score a finetuned Octo checkpoint against the recorded UR10e episodes, starting
from a bare machine. No robot and no simulator needed.

**What "open-loop" means:** we never feed the policy's own actions back to a robot. We
walk a *recorded* episode and every `action_horizon` steps show the policy the recorded
observation, asking it to predict the next chunk of actions. Those predictions are
stitched together and compared to the recorded ground-truth actions. It measures how
well the policy reproduces the demonstrations.

---

## 0. What you need

| | |
|---|---|
| OS | Linux |
| Python | **3.10.12 or newer 3.10.x** (see the 3.10.0 warning in step 2) |
| GPU | Optional. NVIDIA + ~3 GB VRAM makes it fast; CPU works but is slow (tens of minutes/episode) |
| Disk | ~10 GB (venv ~6 GB, checkpoint 0.8 GB, dataset 0.6 GB) |
| Network | Needed once, to download `t5-base` from HuggingFace (see step 4) |

Three things get copied from the training machine: **the code**, **the checkpoint**, and
**the RLDS dataset**.

---

## 1. Get the code

```bash
git clone <this-repo-url> octo-pytorch     # or rsync the repo dir across
cd octo-pytorch
```

---

## 2. Create the virtualenv

> **Do not use Python 3.10.0.** It has an asyncio bug that breaks checkpoint restore
> (`ValueError: loop argument must agree with lock`). Use 3.10.12+.

```bash
python3.10 --version            # confirm >= 3.10.12
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install PyTorch first, matching your machine's CUDA. The training box uses cu130:

```bash
# GPU (CUDA 12.x/13.x -- pick the index-url matching your driver)
pip install torch --index-url https://download.pytorch.org/whl/cu130
# ...or CPU-only:
# pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install accelerate
```

Then the Octo deps, the compatibility pins, and the package itself. **Order matters** --
the pins deliberately downgrade packages that `requirements.txt` pulls in too new:

```bash
pip install -r requirements.txt
pip install -r requirements_compat_pins.txt   # MUST come after requirements.txt
pip install -e .                              # installs the `octo` package
```

Why the pins exist (from `requirements_compat_pins.txt`):
- `transformers==4.34.1` — 5.x removed Flax, which octo's text processor imports.
- `wandb==0.16.6` — >=0.17 ships protobuf-5 stubs that break on protobuf 3.x.
- `tensorflow-metadata==1.14.0` — newer needs protobuf>=5.26, incompatible with TF 2.15.

Sanity check:

```bash
python -c "import torch, tensorflow as tf, tensorflow_datasets, octo; \
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"
```

Reference versions known to work: `torch 2.13.0`, `transformers 4.34.1`,
`tensorflow 2.15.0`, `tensorflow-datasets 4.9.2`, `numpy 1.24.3`, `accelerate 1.14.0`.

---

## 3. Copy the checkpoint and the dataset

### 3a. Checkpoint

On the **training machine**, checkpoints live under:

```
/mnt/data/ur10e-robotiq/octo-small-finetune/checkpoints/octo_finetune/ur10e_pick_cup_<TIMESTAMP>/
├── config.json              <- required
├── dataset_statistics.json  <- required (action mean/std/mask for unnormalizing)
├── example_batch.pickle     <- required (defines expected input shapes)
├── finetune_config.json
├── 2499/  weights.pth       <- one dir per saved step
├── 4999/  weights.pth
└── ...
```

> **Step dirs are named step-1.** The trainer saves at `(i+1) % save_interval == 0` but
> names the dir with `i`. So a 60k run's saves are `2499, 4999, ..., 59999` -- there is
> no `2500` or `60000`. This trips people up.

You need **the four top-level files plus at least one step dir**. Copy the whole run dir
(each step dir is ~730 MB, so grab only the steps you want):

```bash
# on the EVAL machine -- copy run dir but only the final step
RUN=ur10e_pick_cup_<TIMESTAMP>
SRC=<user>@<train-host>:/mnt/data/ur10e-robotiq/octo-small-finetune/checkpoints/octo_finetune/$RUN

mkdir -p ~/octo-eval/$RUN
rsync -av --info=progress2 \
  --include='*/' \
  --include='config.json' --include='dataset_statistics.json' \
  --include='example_batch.pickle' --include='finetune_config.json' \
  --include='59999/***' \
  --exclude='*' \
  "$SRC/" ~/octo-eval/$RUN/
```

To grab **every** checkpoint instead (for a MSE-vs-steps curve), just:
`rsync -av "$SRC/" ~/octo-eval/$RUN/` (~18 GB for 24 steps).

### 3b. Dataset

The eval reads the recorded episodes it scores against, in RLDS/TFDS form (594 MB):

```bash
rsync -av --info=progress2 \
  <user>@<train-host>:/home/khanh/work/octo-pytorch/rlds/ur10e_pick_cup \
  ~/octo-eval/rlds/
```

Result: `~/octo-eval/rlds/ur10e_pick_cup/1.0.0/*.tfrecord-*`.

> Keep the dataset on an **SSD** — it's read repeatedly.

---

## 4. HuggingFace `t5-base`

The model builds a frozen T5 language encoder at load time
(`T5EncoderModel.from_pretrained("google-t5/t5-base")`), and the tokenizer is `t5-base`.
On a networked machine this just downloads (~900 MB) on first run. Nothing to do.

**Offline?** Copy the cache from the training machine:

```bash
rsync -av <user>@<train-host>:~/.cache/huggingface/hub/models--google-t5--t5-base ~/.cache/huggingface/hub/
rsync -av <user>@<train-host>:~/.cache/huggingface/hub/models--t5-base            ~/.cache/huggingface/hub/
export HF_HUB_OFFLINE=1
```

---

## 5. Run the eval

```bash
cd octo-pytorch
source .venv/bin/activate

python open_loop_eval.py \
  --checkpoint ~/octo-eval/ur10e_pick_cup_<TIMESTAMP> \
  --step 59999 \
  --data-dir ~/octo-eval/rlds \
  --dataset-name ur10e_pick_cup \
  --episodes 5 \
  --out eval_out
```

| flag | meaning |
|---|---|
| `--checkpoint` | the **run dir**, not the step dir |
| `--step` | which step to load; omit for the latest |
| `--data-dir` | the TFDS root (the dir *containing* `ur10e_pick_cup/`) |
| `--episodes` | how many episodes to score (default 3) |
| `--action-horizon` | actions per policy query (default 4 — matches training; leave it) |
| `--device` | `cuda:0` (default) or `cpu` |
| `--out` | where the per-episode PNG plots go |

Outputs: per-dimension MSE/MAE printed per episode and averaged, plus one
`eval_out/episode_NNN.png` per episode showing ground truth (solid) vs prediction
(dashed) for each action dimension.

---

## 6. How to read the numbers

Action layout: `[dx, dy, dz, droll, dpitch, dyaw, gripper]`, deltas in metres/radians,
gripper absolute binary (1=open).

**Compare against these trivial baselines**, measured on this dataset. A policy that
doesn't beat them has learned nothing:

| dim | always-predict-0 MSE | always-predict-mean MSE |
|---|---|---|
| dx | 9.80e-06 | 7.98e-06 |
| dy | 3.24e-06 | 2.99e-06 |
| dz | 1.32e-05 | 1.31e-05 |
| droll | 8.76e-09 | 8.67e-09 |
| dpitch | 1.16e-09 | 1.15e-09 |
| dyaw | 3.23e-09 | 3.23e-09 |
| **gripper** | 4.96e-01 | **2.50e-01** |

Two things to keep in mind:

**Ignore the rotation dims.** The wrist is held fixed throughout this task
(ground-truth std ~1e-4 rad), so `droll`/`dpitch`/`dyaw` score near-zero MSE trivially
by predicting nothing. They tell you nothing about policy quality. **Judge on `dx`,
`dy`, `dz`, and `gripper`.**

**The bar is beating the baseline column.** `dx` MSE must come in under ~8e-6, `dz`
under ~1.3e-5, gripper under 0.25. For reference, an early checkpoint (step 2499) scored
`dx=4.2e-5`, `dz=1.3e-4`, `gripper=0.38` — i.e. **worse than predicting zero**, which is
what under-training looks like.

**Also look at the plots**, not just the numbers. In `episode_NNN.png`, the thing to
check is whether the gripper trace flips from open to closed at roughly the same
timestep as ground truth. A policy can have a decent MSE while grasping at the wrong
moment, which fails on a real robot.

**The training loss is not a quality signal.** Octo-small uses a diffusion action head
whose loss samples a random noise level each batch; it sits near a floor (~1.5) and
barely moves even while the policy improves a lot. Use this eval, not the loss.

---

## 7. Troubleshooting

**`FileNotFoundError: .../example_batch.pickle`** — you pointed `--checkpoint` at a step
dir, or didn't copy the four top-level files. Point it at the **run dir**.

**`FileNotFoundError: 'hf:/rail-berkeley/...'`** — `open_loop_eval.py` only accepts a
local checkpoint dir; it can't take an `hf://` path.

**No `59999` dir** — remember saves are named step-1 (`2499`, `4999`, ...). List the run
dir and use what's there.

**CUDA out of memory** — the eval needs ~2-3 GB. If something else holds the GPU, either
free it or run with `--device cpu`.

**TensorFlow grabbing the GPU** — already handled: the script calls
`tf.config.set_visible_devices([], "GPU")` so TF stays on CPU and the policy gets the VRAM.

**Noisy TF logs** (`Error in PredictCost()`, cuDNN factory warnings) — harmless. Silence
with `export TF_CPP_MIN_LOG_LEVEL=3`.

**Predictions look ~2x too wide** — partly expected: the head is a diffusion model and
the script draws a *single* stochastic sample per query rather than averaging, so even a
perfect policy shows extra spread.
