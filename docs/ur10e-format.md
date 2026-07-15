# UR10e Teleop Dataset Format

Raw teleoperation episodes recorded from a real UR10e + Robotiq 2F gripper,
driven with a Logitech F710 gamepad (see [COLLECT_DATA.md](COLLECT_DATA.md)).
Each episode is one HDF5 file written by
[data_recorder.py](src/vr_ur_teleop/vr_ur_teleop/data_recorder.py) into `episodes/`.

## File layout

```
episodes/
├── episode_000000.hdf5
├── episode_000001.hdf5
└── ...
```

Episode index = number of existing `episode_*.hdf5` files at save time (0-based, 6 digits).

## HDF5 structure

`T` = number of timesteps in the episode.

```
episode_XXXXXX.hdf5
├── observations/
│   ├── images/
│   │   ├── side    [T, 480, 640, 3]  uint8    RGB, D455 side camera
│   │   └── wrist   [T, 480, 640, 3]  uint8    RGB, D435I wrist camera
│   ├── arm_joint_pos   [T, 6]  float32   joint angles, rad
│   └── gripper_pos     [T, 1]  float32   0.0=closed, 1.0=open
├── actions/
│   ├── arm_joint_pos   [T, 6]  float32   target joint angles, rad
│   └── gripper_pos     [T, 1]  float32   target gripper state
└── metadata/           (attrs only, no datasets)
    ├── task_description  str    e.g. "pick up the cup"
    ├── success           bool   set via /data_recorder/set_success
    ├── episode_type      str    "full_vla" | "sphere"
    ├── fps               int    declared recording rate (20 in current data)
    └── n_timesteps       int    = T
```

All datasets are contiguous and uncompressed, so files are large
(~100 MB - 1.3 GB depending on length).

## Field semantics

### observations/images/{side,wrist}

- RGB (not BGR), uint8, 480x640.
- Frames from the two RealSense cameras are paired with
  `ApproximateTimeSynchronizer` (slop 0.05 s). One timestep is appended per
  synced pair, so the real capture rate follows the camera sync rate, not the
  `fps` metadata value. Images are resized to 640x480 if the stream differs.

### observations/arm_joint_pos

- Measured joint angles from RTDE `getActualQ()` (published at 60 Hz on
  `/ur/joint_states`); the recorder stores the latest value at each camera pair.
- Joint order (standard UR):
  1. shoulder_pan_joint
  2. shoulder_lift_joint
  3. elbow_joint
  4. wrist_1_joint
  5. wrist_2_joint
  6. wrist_3_joint

### observations/gripper_pos

- Binary commanded state, not a measured width: 0.0 = closed, 1.0 = open.
- Comes from `/ur/gripper/position` (10 Hz), which mirrors the last
  open/close command (RB toggle on the gamepad).

### actions/*

- Actions are the observations shifted by one step:
  `action[t] = observation[t+1]`, and the last action is duplicated
  (`action[T-1] = observation[T-1]`).
- So `actions/arm_joint_pos` is an absolute joint-position target, and
  `actions/gripper_pos` is the binary gripper target. There is no separate
  recorded command stream.

## Timing notes

- `metadata/fps` is the `fps` parameter passed to the recorder
  (`run_service_record_data.sh` sets 20). It is a nominal value; actual step
  spacing is set by the camera pair rate (cameras run at 30 fps).
- Joints and gripper are sampled asynchronously (latest-value hold), so each
  timestep can lag the image pair by up to their publish period
  (~17 ms joints, ~100 ms gripper).

## Current data (as of 2026-07-10)

- 13 episodes (`episode_000000` - `episode_000012`), 8392 frames total,
  T ranges from 53 to 3225.
- All: task "pick up the cup", `episode_type=full_vla`, `success=True`, fps=20.

## Reading episodes

```bash
# Summary (shapes, metadata, joint stats)
python3 scripts/read_episode.py episodes/episode_000012.hdf5

# Export videos (one .mp4 per camera)
python3 scripts/read_episode.py episodes/episode_000012.hdf5 --video --fps 20 --out ep012_video

# Plot joints + gripper -> episode_plot.png
python3 scripts/read_episode.py episodes/episode_000012.hdf5 --plot

# Export frames as PNGs
python3 scripts/read_episode.py episodes/episode_000012.hdf5 --save-frames all --out frames_dir
```

## Downstream converters

- [convert_ur_to_libero_lerobot.py](convert_ur_to_libero_lerobot.py) -
  LeRobot v2.1 dataset in the LIBERO layout (absolute EEF state, delta EEF
  action, EEF pose from UR10e forward kinematics).
- [convert_ur_to_groot_n1d7.py](convert_ur_to_groot_n1d7.py) -
  GR00T N1.7 relative-EEF LeRobot v2 dataset (state 16-d, action 10-d,
  gripper remapped to {+1 open, -1 close}).

Both rely on the `action[t] == observation[t+1]` convention above to compute
target EEF poses via forward kinematics.
