# Octo Dataset Format (RLDS)

What the Octo repos at `/home/khanhnd61/work/octo` (JAX) and
`/home/khanhnd61/work/octo-pytorch` (PyTorch port) expect as training data,
and how to get the UR10e episodes (see [ur10e-format.md](ur10e-format.md)) into it.

**Key fact: both repos use the exact same data pipeline and format.**
`octo/data/` is byte-identical between the two repos (verified with `diff -rq`).
octo-pytorch just wraps the TensorFlow data iterator in a
`TorchRLDSDataset` (`octo/utils/torch_rlds_dataset.py`) that converts numpy ->
torch tensors (and HWC -> CHW). So there is one format: **RLDS**
(Reinforcement Learning Datasets), which is a TFDS (TensorFlow Datasets)
dataset of episodes. There is no separate "octo-pytorch format".

## On-disk layout

An RLDS dataset is a built TFDS dataset directory:

```
<data_dir>/
└── <dataset_name>/          # e.g. ur10e_pick_cup
    └── 1.0.0/
        ├── dataset_info.json
        ├── features.json
        ├── <dataset_name>-train.tfrecord-00000-of-000NN
        └── ...
```

Loaded with `tfds.builder(name, data_dir=...)` in
`make_dataset_from_rlds()` ([octo/data/dataset.py](../../octo/octo/data/dataset.py)).
Custom datasets are normally created with the
[rlds_dataset_builder](https://github.com/kpertsch/rlds_dataset_builder)
template: you write a `tfds.core.GeneratorBasedBuilder` whose
`_generate_examples()` reads your raw files (our HDF5 episodes) and yields
episodes, then run `tfds build`.

If the dataset has no `val` split, the loader auto-splits:
`train[:95%]` for training, `train[95%:]` for validation.

## Episode / step structure

Each episode is a dict with a `steps` sequence. Standard RLDS step layout
(what the rlds_dataset_builder template produces):

```
episode = {
  "steps": [                       # one entry per timestep
    {
      "observation": {
        "image":       Image(H, W, 3) uint8    # third-person cam (any name)
        "wrist_image": Image(H, W, 3) uint8    # wrist cam (any name)
        "state":       Tensor(D,) float32      # proprio (any name)
      },
      "action":       Tensor(A,) float32,
      "language_instruction": Text,
      "discount":     float32,                 # 1.0
      "reward":       float32,                 # 1.0 on last step
      "is_first":     bool,
      "is_last":      bool,
      "is_terminal":  bool,
    },
    ...
  ],
  "episode_metadata": {"file_path": Text},
}
```

Octo's loader only actually uses `observation`, `action`, and the language
key. Observation key names are free-form: they are remapped at load time via
`image_obs_keys={"primary": "image", "wrist": "wrist_image"}`,
`proprio_obs_key="state"`, `language_key="language_instruction"`.
Images are stored JPEG-encoded by TFDS (`tfds.features.Image`) and only
decoded + resized in the frame transforms, so files are much smaller than raw
HDF5.

## What the loader produces

Pipeline (`make_single_dataset` / `make_interleaved_dataset`):

1. `make_dataset_from_rlds`: optional `standardize_fn` per trajectory ->
   rename image/proprio keys -> extract `task["language_instruction"]` ->
   normalize action and proprio (see below).
2. `apply_trajectory_transforms`: goal relabeling, padding masks, and
   chunking (`chunk_act_obs`): observations get a history axis of
   `window_size`, actions get `[window_size, action_horizon]` axes
   (current + future actions).
3. `apply_frame_transforms`: decode + resize images, augmentation.

A training batch looks like (window_size=2, action_horizon=4, action_dim=7):

```
batch = {
  "observation": {
    "image_primary":     [B, 2, 256, 256, 3] uint8
    "image_wrist":       [B, 2, 128, 128, 3] uint8
    "timestep":          [B, 2] int32
    "timestep_pad_mask": [B, 2] bool
    "pad_mask_dict":     per-key bool masks
    ("proprio":          [B, 2, D] float32, only if proprio_obs_key set)
  },
  "task": {
    "language_instruction": tokenized (input_ids, attention_mask)
    (+ goal images if goal relabeling is on)
  },
  "action":          [B, 2, 4, 7] float32, normalized
  "action_pad_mask": [B, 2, 4, 7] bool
}
```

### Normalization

- Actions and proprio are normalized with per-dataset statistics
  (`mean/std` for "normal", `p01/p99` for "bounds"). Statistics are computed
  automatically on first load and cached, or passed as a JSON file.
- `action_normalization_mask` excludes dimensions from normalization;
  standard for a 7-dim EEF action: `[True]*6 + [False]` so the binary
  gripper dim stays 0/1.

## What pretrained octo-small-1.5 expects

From `/home/khanhnd61/work/octo-ref/octo-small-1.5/config.json`:

- Observations: `image_primary` (third-person, resized 256x256) and
  `image_wrist` (128x128). No proprio input.
- Task: language (t5-base tokenizer) and/or goal image.
- `window_size=2`, diffusion action head with `action_horizon=4`,
  `action_dim=7`.
- Action convention (bridge/OXE `EEF_POS`):
  `[dx, dy, dz, droll, dpitch, dyaw, gripper]`
  - deltas of the EEF pose in the robot base/world frame between
    consecutive steps (bridge relabels action = state[t+1] - state[t])
  - gripper is absolute and binary: 1=open, 0=close, not normalized
- Normalization type: "normal" (mean 0, std 1) per dataset.

Finetuning can change the observation and action space entirely (see
`examples/02_finetune_new_observation_action.py`: adds proprio tokenizer, new
L1 head with 14-dim actions), but then the action head is trained from
scratch. Keeping the 7-dim delta-EEF convention reuses the pretrained head.

## octo vs octo-pytorch

| | octo (JAX) | octo-pytorch |
|---|---|---|
| Data format | RLDS/TFDS | identical (same `octo/data/` code, TF-based) |
| Loader entry | `make_single_dataset` -> `.iterator()` | `make_single_dataset` -> `TorchRLDSDataset` -> `DataLoader` |
| Finetune script | `scripts/finetune.py` | `scripts/finetune_pt.py` |
| Weights | native | can load JAX checkpoints (`OctoModelPt.load_pretrained_from_jax`) |

Both need TensorFlow + `tensorflow_datasets` + `dlimp` at training time for
the data pipeline (even the pytorch one).

## Converting the UR10e episodes -> answer

**Convert to RLDS. One conversion serves both octo and octo-pytorch.**
Do not use the LeRobot or GR00T N1.7 outputs of `convert_ur_to_*.py` for
Octo: Octo's loader reads only TFDS/RLDS; it cannot read LeRobot parquet+mp4
or the GR00T layout. Those converters stay useful for their own stacks.

Recommended mapping (a new `convert_ur_to_rlds.py`, reusing the FK and
delta-EEF code already in
[convert_ur_to_libero_lerobot.py](convert_ur_to_libero_lerobot.py)):

| UR10e HDF5 | RLDS step |
|---|---|
| `observations/images/side [T,480,640,3]` | `observation/image` (store 256x256, JPEG) -> `image_primary` |
| `observations/images/wrist [T,480,640,3]` | `observation/wrist_image` (store 128x128 or 256x256) -> `image_wrist` |
| `observations/arm_joint_pos [T,6]` + FK | `observation/state` (e.g. EEF xyz+rpy+gripper, optional) |
| FK(obs joints): pose[t+1] - pose[t] | `action[0:6]` = world-frame delta EEF |
| `actions/gripper_pos [T,1]` | `action[6]` gripper, already 1=open / 0=closed - matches bridge convention |
| `metadata/task_description` | `language_instruction` (every step) |

Notes:

- The UR data uses `action[t] = observation[t+1]`, so
  `action_eef[t] = FK(obs_joint[t+1]) - FK(obs_joint[t])` and the last step
  should be dropped (same as bridge's `relabel_actions`).
- Frame rate: episodes step at the camera-pair rate (~20-30 Hz), while
  bridge-style pretraining data is ~5 Hz. Per-step deltas at 30 Hz are tiny;
  consider subsampling (e.g. every 4th-6th frame) during conversion so
  action magnitudes are closer to pretraining and horizons cover more time.
- Rotation delta: use rpy (roll, pitch, yaw) deltas to match `EEF_POS`
  encoding, not axis-angle (the LIBERO converter uses axis-angle; change
  this when reusing the code).
- Let the loader compute dataset statistics; pass
  `action_normalization_mask=[True]*6+[False]`.
- Loader kwargs for finetuning:

```python
dataset_kwargs = dict(
    name="ur10e_pick_cup",
    data_dir="<data_dir>",
    image_obs_keys={"primary": "image", "wrist": "wrist_image"},
    language_key="language_instruction",
    action_proprio_normalization_type="normal",
    action_normalization_mask=[True]*6 + [False],
)
traj_transform_kwargs = dict(window_size=2, action_horizon=4)
frame_transform_kwargs = dict(resize_size={"primary": (256, 256), "wrist": (128, 128)})
```

Alternative: finetune with absolute joint actions (6 joints + gripper)
instead of delta EEF. The pipeline supports it (it is just a different 7-dim
vector), but the pretrained action head no longer matches, so it gets
re-initialized - only worth it if delta-EEF control is not an option at
deployment.
