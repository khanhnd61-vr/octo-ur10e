#!/usr/bin/env python3
"""
convert_ur_to_rlds.py

Convert raw UR10e teleop episodes (episodes/*.hdf5, see ur10e-format.md) into an
RLDS (TFDS) dataset that Octo (JAX) and octo-pytorch can load directly
(see octo-format.md).

Output: <out>/ur10e_pick_cup/1.0.0/ with tfrecord shards. The dataset name comes
from the builder class name (Ur10ePickCup -> ur10e_pick_cup); edit the class
name to change it.

Per-step layout (bridge/OXE EEF_POS convention):
  observation/image        [256,256,3] uint8   side cam (jpeg-encoded by TFDS)
  observation/wrist_image  [128,128,3] uint8   wrist cam
  (sizes are --primary-size / --wrist-size; defaults match what the
   octo-small-1.5 tokenizers were trained on. --img-mode resize squashes
   640x480 to square like OXE preprocessing; crop center-crops first,
   no distortion but drops the side FOV.)
  observation/state        (7,)  float32   [x,y,z, roll,pitch,yaw, gripper] abs EEF pose
  observation/joint_state  (6,)  float32   UR joint angles, rad
  action                   (7,)  float32   [dx,dy,dz, droll,dpitch,dyaw, gripper]
                                           world-frame delta EEF between kept frames,
                                           gripper absolute binary (1=open, 0=close)
  language_instruction     str             from metadata/task_description

EEF pose = UR10e forward kinematics (standard DH) of the recorded joints.
Frames are subsampled with --stride (recorder steps at the ~20-30 Hz camera
rate; bridge-style pretraining data is ~5 Hz). The last kept frame is dropped
(no next pose to compute a delta), same as bridge's relabel_actions.

Octo loader kwargs for the result:
  dataset_kwargs = dict(
      name="ur10e_pick_cup",
      data_dir="<out>",
      image_obs_keys={"primary": "image", "wrist": "wrist_image"},
      language_key="language_instruction",
      action_proprio_normalization_type="normal",
      action_normalization_mask=[True]*6 + [False],
  )

Run (env from requirements_rlds.txt):
  python3 convert_ur_to_rlds.py --src episodes --out rlds --stride 5
"""
import argparse
import glob
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

try:
    import cv2
    import h5py
    import tensorflow_datasets as tfds
    from scipy.spatial.transform import Rotation
except ImportError as e:
    sys.exit(f"Missing dep: {e}. Install with: pip install -r requirements_rlds.txt")

# TFDS otherwise probes gs://tfds-data on builder init to fetch canned dataset
# info; TF's C++ GCS filesystem plugin segfaults on that in offline/sandboxed
# environments. We build a local custom dataset, so disable GCS entirely.
try:
    tfds.core.utils.gcs_utils._is_gcs_disabled = True
except Exception:
    pass
os.environ.setdefault("NO_GCE_CHECK", "true")


# --------------------------------------------------------------------------- #
#  UR10e forward kinematics (official e-Series standard DH parameters)
# --------------------------------------------------------------------------- #
UR10E_DH_A     = np.array([0.0,     -0.6127, -0.57155, 0.0,     0.0,      0.0])
UR10E_DH_D     = np.array([0.1807,   0.0,     0.0,     0.17415, 0.11985,  0.11655])
UR10E_DH_ALPHA = np.array([np.pi/2,  0.0,     0.0,     np.pi/2, -np.pi/2, 0.0])


def _dh_T(theta, a, d, alpha):
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0,      sa,       ca,      d],
        [0.0,     0.0,      0.0,    1.0],
    ])


def ur10e_fk(joints, tcp_z_offset=0.0):
    """joints (T,6) rad -> pos (T,3), rotmat (T,3,3) of the flange in base frame."""
    T = len(joints)
    pos = np.zeros((T, 3))
    rot = np.zeros((T, 3, 3))
    tcp = np.eye(4)
    tcp[2, 3] = tcp_z_offset
    for i in range(T):
        M = np.eye(4)
        for j in range(6):
            M = M @ _dh_T(joints[i, j], UR10E_DH_A[j], UR10E_DH_D[j], UR10E_DH_ALPHA[j])
        M = M @ tcp
        pos[i] = M[:3, 3]
        rot[i] = M[:3, :3]
    return pos, rot


def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))


def to_square(img, size, mode):
    """HxWx3 uint8 RGB -> size x size x3. mode: 'resize' (squash) or 'crop' (center square)."""
    if mode == "crop":
        h, w = img.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        img = img[y0:y0 + s, x0:x0 + s]
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------- #
#  TFDS builder
# --------------------------------------------------------------------------- #
class Ur10ePickCup(tfds.core.GeneratorBasedBuilder):
    """UR10e teleop pick-and-place episodes in RLDS format."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial conversion from vr_ur_teleop HDF5 episodes."}

    def __init__(self, *, src_files, stride, primary_size, wrist_size, img_mode,
                 language, tcp_z_offset, **kwargs):
        self._src_files = src_files
        self._stride = stride
        self._primary_size = primary_size
        self._wrist_size = wrist_size
        self._img_mode = img_mode
        self._language = language
        self._tcp_z_offset = tcp_z_offset
        super().__init__(**kwargs)

    def _info(self):
        P, W = self._primary_size, self._wrist_size
        return tfds.core.DatasetInfo(
            builder=self,
            description="UR10e + Robotiq 2F teleop episodes (vr_ur_teleop).",
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(
                            shape=(P, P, 3), dtype=np.uint8, encoding_format="jpeg",
                            doc="Side camera (D455) RGB.",
                        ),
                        "wrist_image": tfds.features.Image(
                            shape=(W, W, 3), dtype=np.uint8, encoding_format="jpeg",
                            doc="Wrist camera (D435I) RGB.",
                        ),
                        "state": tfds.features.Tensor(
                            shape=(7,), dtype=np.float32,
                            doc="Absolute EEF pose [x,y,z, roll,pitch,yaw] + gripper (1=open).",
                        ),
                        "joint_state": tfds.features.Tensor(
                            shape=(6,), dtype=np.float32,
                            doc="UR joint angles, rad (shoulder_pan..wrist_3).",
                        ),
                    }),
                    "action": tfds.features.Tensor(
                        shape=(7,), dtype=np.float32,
                        doc="World-frame delta EEF [dx,dy,dz, droll,dpitch,dyaw] + gripper (1=open).",
                    ),
                    "discount": tfds.features.Scalar(dtype=np.float32),
                    "reward": tfds.features.Scalar(dtype=np.float32),
                    "is_first": np.bool_,
                    "is_last": np.bool_,
                    "is_terminal": np.bool_,
                    "language_instruction": tfds.features.Text(),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(),
                }),
            }),
        )

    def _split_generators(self, dl_manager):
        return {"train": self._generate_examples()}

    def _generate_examples(self):
        for path in self._src_files:
            episode = self._convert_episode(path)
            if episode is not None:
                yield os.path.basename(path), episode

    def _convert_episode(self, path):
        with h5py.File(path, "r") as f:
            meta = dict(f["metadata"].attrs)
            if not bool(meta.get("success", True)):
                print(f"skip (success=False): {path}")
                return None

            joints = f["observations/arm_joint_pos"][:]          # (T,6)
            grip = f["observations/gripper_pos"][:, 0]           # (T,)
            idx = np.arange(0, len(joints), self._stride)
            if len(idx) < 2:
                print(f"skip (too short after stride): {path}")
                return None

            lang = self._language or str(meta.get("task_description", ""))

            pos, rot = ur10e_fk(joints[idx], self._tcp_z_offset)   # (K,3), (K,3,3)
            rpy = Rotation.from_matrix(rot).as_euler("xyz")        # (K,3)
            state = np.concatenate([pos, rpy, grip[idx, None]], axis=1).astype(np.float32)

            dpos = pos[1:] - pos[:-1]
            drpy = wrap_angle(rpy[1:] - rpy[:-1])
            # gripper action = gripper state at the next kept frame (the target),
            # consistent with the recorder's action[t] = observation[t+1]
            g_act = grip[idx[1:], None]
            action = np.concatenate([dpos, drpy, g_act], axis=1).astype(np.float32)

            side = f["observations/images/side"]
            wrist = f["observations/images/wrist"]
            n = len(idx) - 1  # last kept frame dropped (no next pose for a delta)
            steps = []
            for j in range(n):
                t = int(idx[j])
                steps.append({
                    "observation": {
                        "image": to_square(side[t], self._primary_size, self._img_mode),
                        "wrist_image": to_square(wrist[t], self._wrist_size, self._img_mode),
                        "state": state[j],
                        "joint_state": joints[t].astype(np.float32),
                    },
                    "action": action[j],
                    "discount": np.float32(1.0),
                    "reward": np.float32(j == n - 1),
                    "is_first": j == 0,
                    "is_last": j == n - 1,
                    "is_terminal": j == n - 1,
                    "language_instruction": lang,
                })
        print(f"{os.path.basename(path)}: {len(joints)} frames -> {n} steps (stride {self._stride})")
        return {"steps": steps, "episode_metadata": {"file_path": path}}


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default="episodes", help="Dir with episode_*.hdf5 (default: episodes)")
    p.add_argument("--out", default="rlds", help="TFDS data_dir to write into (default: rlds)")
    p.add_argument("--stride", type=int, default=5,
                   help="Keep every Nth frame (default: 5, ~30Hz -> ~6Hz)")
    p.add_argument("--primary-size", type=int, default=256,
                   help="Stored side-cam size (default: 256, octo image_primary input size)")
    p.add_argument("--wrist-size", type=int, default=128,
                   help="Stored wrist-cam size (default: 128, octo image_wrist input size)")
    p.add_argument("--img-mode", choices=["resize", "crop"], default="resize",
                   help="resize: squash 640x480 to square (matches OXE preprocessing); crop: center-crop first")
    p.add_argument("--language", default=None,
                   help="Override language instruction (default: metadata/task_description)")
    p.add_argument("--tcp-z-offset", type=float, default=0.0,
                   help="Tool z offset added to flange FK, meters (default: 0)")
    p.add_argument("--min-id", type=int, default=0,
                   help="Lowest episode id to include, inclusive (default: 0)")
    p.add_argument("--max-id", type=int, default=63,
                   help="Highest episode id to include, inclusive (default: 63)")
    args = p.parse_args()

    all_files = sorted(glob.glob(os.path.join(args.src, "episode_*.hdf5")))
    if not all_files:
        sys.exit(f"No episode_*.hdf5 found in {args.src}")

    def episode_id(path):
        return int(os.path.basename(path)[len("episode_"):-len(".hdf5")])

    src_files = [f for f in all_files if args.min_id <= episode_id(f) <= args.max_id]
    skipped = [os.path.basename(f) for f in all_files if f not in src_files]
    if not src_files:
        sys.exit(f"No episodes in id range [{args.min_id}, {args.max_id}] in {args.src}")
    print(f"Found {len(all_files)} episodes in {args.src}; "
          f"using {len(src_files)} in id range [{args.min_id}, {args.max_id}]")
    if skipped:
        print(f"Excluded {len(skipped)} out-of-range: {', '.join(skipped)}")

    builder = Ur10ePickCup(
        src_files=src_files,
        stride=args.stride,
        primary_size=args.primary_size,
        wrist_size=args.wrist_size,
        img_mode=args.img_mode,
        language=args.language,
        tcp_z_offset=args.tcp_z_offset,
        data_dir=args.out,
    )
    builder.download_and_prepare()
    print(f"\nDone -> {builder.data_dir}")
    print("If the dataset already existed at this path, nothing was rewritten; "
          "delete the dir or bump VERSION to rebuild.")
    print(f"""
Octo loader kwargs:
  dataset_kwargs = dict(
      name="{builder.name}",
      data_dir="{os.path.abspath(args.out)}",
      image_obs_keys={{"primary": "image", "wrist": "wrist_image"}},
      language_key="language_instruction",
      action_proprio_normalization_type="normal",
      action_normalization_mask=[True]*6 + [False],
  )""")


if __name__ == "__main__":
    main()
