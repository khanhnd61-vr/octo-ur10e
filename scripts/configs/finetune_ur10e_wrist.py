"""Finetune config for UR10e pick-cup using the WRIST CAMERA ONLY (v1.1.1).

Same as finetune_ur10e.py but drops the 3rd-person (primary/side) view entirely.

Two changes are required -- doing only the first is a common mistake:

  1. dataset: image_obs_keys = {"wrist": "wrist_image"}  (no "primary" entry).
     Note that mapping "primary" to None does NOT remove it -- per the octo docs a
     None value *inserts a padding image*, so the primary tokenizer would still run
     on a blank image and still emit its 256 tokens. Omitting the key entirely means
     the 256x256 side images are never even decoded (a large RAM saving).

  2. model: delete the `primary` observation tokenizer via config_delete_keys, which
     finetune_pt.py applies to the pretrained config before building the model. This
     is the equivalent of the JAX example's
     `del config["model"]["observation_tokenizers"]["wrist"]`.

Token budget per timestep (octo-small-1.5, stem-16 patches):
    primary 256x256 -> 16x16 = 256 tokens
    wrist   128x128 ->  8x8  =  64 tokens
  both views = 320 visual tokens; wrist-only = 64  ->  a 5x (-80%) reduction.

The wrist tokenizer and transformer keep their pretrained weights; only the primary
tokenizer's weights go unused. octo-small-1.5 was pretrained with both views, so
wrist-only is a distribution shift the finetune has to absorb.

Usage:
  torchrun --nproc_per_node 1 scripts/finetune_pt.py \
    --config scripts/configs/finetune_ur10e_wrist.py:full,language_conditioned \
    --config.pretrained_path=hf://rail-berkeley/octo-small-1.5
"""
from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder

DATA_DIR = "/home/khanh/work/octo-pytorch/rlds"


def get_config(config_string="full,language_conditioned"):
    mode, task = config_string.split(",")
    assert task in ["image_conditioned", "language_conditioned", "multimodal"]
    assert mode in ["full", "head_only", "head_mlp_only"]

    FINETUNING_KWARGS = {
        "name": "ur10e_pick_cup",
        "data_dir": DATA_DIR,
        # WRIST ONLY -- "primary" is omitted, not set to None (None would pad, not drop).
        "image_obs_keys": {"wrist": "wrist_image"},
        "proprio_obs_key": None,
        "language_key": "language_instruction",
        "action_proprio_normalization_type": "normal",
        # delta-EEF 6 dims normalized, binary gripper (dim 6) left as 0/1.
        # (See finetune_ur10e.py for why the rotation dims must stay masked-True.)
        "action_normalization_mask": [True, True, True, True, True, True, False],
        "standardize_fn": None,
    }

    if mode == "full":
        frozen_keys = None
    elif mode == "head_only":
        frozen_keys = ("octo_transformer.*",)
    elif mode == "head_mlp_only":
        frozen_keys = (
            "octo_transformer.*",
            "heads_*.map_head.probe",
            "heads_*.map_head.MultiHeadDotProductAttention_0.*",
        )
    else:
        raise ValueError("Invalid mode")

    max_steps = FieldReference(60000)
    window_size = FieldReference(default=2)

    config = dict(
        pretrained_path=placeholder(str),
        pretrained_step=placeholder(int),
        batch_size=16,
        shuffle_buffer_size=1000,
        num_steps=max_steps,
        log_interval=100,
        eval_interval=2500,
        save_interval=2500,
        save_dir=placeholder(str),
        seed=42,
        wandb=dict(
            project="octo_finetune", group=placeholder(str), entity=placeholder(str)
        ),
        dataset_kwargs=FINETUNING_KWARGS,
        modality=task,
        finetuning_mode=mode,
        window_size=window_size,
        # Remove the primary (side-cam) observation tokenizer from the pretrained
        # model config. finetune_pt.py deletes every flattened config key that starts
        # with this path, before the model is constructed.
        config_delete_keys=dict(
            model=dict(observation_tokenizers=dict(primary=None)),
        ),
        optimizer=dict(
            learning_rate=dict(
                name="cosine",
                init_value=0.0,
                peak_value=3e-4,
                warmup_steps=2000,
                decay_steps=max_steps,
                end_value=0.0,
            ),
            weight_decay=0.01,
            clip_gradient=1.0,
            frozen_keys=frozen_keys,
            grad_accumulation_steps=None,
        ),
        val_kwargs=dict(
            val_shuffle_buffer_size=1000,
            num_val_batches=8,
        ),
        viz_kwargs=dict(
            eval_batch_size=8,
            trajs_for_metrics=100,
            trajs_for_viz=8,
            samples_per_state=8,
        ),
    )

    if task == "image_conditioned":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 1.0
    elif task == "language_conditioned":
        goal_relabeling_strategy = None
        keep_image_prob = 0.0
    elif task == "multimodal":
        goal_relabeling_strategy = "uniform"
        keep_image_prob = 0.5
    else:
        raise ValueError("Invalid modality")

    traj_transform_kwargs = dict(
        window_size=window_size,
        action_horizon=4,
        goal_relabeling_strategy=goal_relabeling_strategy,
        task_augment_strategy="delete_task_conditioning",
        task_augment_kwargs=dict(
            keep_image_prob=keep_image_prob,
        ),
    )
    # Only the wrist stream survives -> only wrist augmentation/resize is defined.
    wrist_augment_kwargs = dict(
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
    frame_transform_kwargs = dict(
        resize_size={
            "wrist": (128, 128),
        },
        image_augment_kwargs=dict(
            wrist=wrist_augment_kwargs,
        ),
    )
    config["frame_transform_threads"] = 16
    config["traj_transform_kwargs"] = traj_transform_kwargs
    config["frame_transform_kwargs"] = frame_transform_kwargs
    return ConfigDict(config)
