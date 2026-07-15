"""Finetune config for the UR10e pick-cup RLDS dataset (delta-EEF, 7-dim).

Derived from finetune_config.py. Key differences:
  * points at the ur10e_pick_cup RLDS dataset produced by convert_ur_to_rlds.py
  * standardize_fn=None  -- the converter already emits octo-standard keys
    (image / wrist_image / state / language_instruction) and delta-EEF actions
    with a binary gripper, so NO bridge-style relabel_actions is wanted here.
  * no proprio input (octo-small-1.5 has no proprio tokenizer; keep the
    pretrained architecture so the diffusion action head is reused).
  * window_size=2 to match octo-small-1.5.
  * batch/eval sizes shrunk to fit a single 12 GB GPU (RTX 3060).

Usage:
  torchrun --nproc_per_node 1 scripts/finetune_pt.py \
    --config scripts/configs/finetune_ur10e.py:full,language_conditioned \
    --config.pretrained_path=hf://rail-berkeley/octo-small-1.5
"""
from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder

# Absolute path to the RLDS output dir written by convert_ur_to_rlds.py --out rlds
DATA_DIR = "/home/khanh/work/octo-pytorch/rlds"


def get_config(config_string="full,language_conditioned"):
    mode, task = config_string.split(",")
    assert task in ["image_conditioned", "language_conditioned", "multimodal"]
    assert mode in ["full", "head_only", "head_mlp_only"]

    FINETUNING_KWARGS = {
        "name": "ur10e_pick_cup",
        "data_dir": DATA_DIR,
        # primary = 3rd-person side cam, wrist = wrist cam
        "image_obs_keys": {"primary": "image", "wrist": "wrist_image"},
        # octo-small-1.5 takes no proprio input -> keep it out so the
        # pretrained architecture/head is reused unchanged.
        "proprio_obs_key": None,
        "language_key": "language_instruction",
        "action_proprio_normalization_type": "normal",
        # delta-EEF 6 dims normalized, binary gripper (dim 6) left as 0/1.
        #
        # Do NOT unmask the rotation dims. The wrist is held fixed here, so
        # droll/dpitch/dyaw have std ~1e-4; it is tempting to skip normalizing
        # them so that near-zero std doesn't inflate FK jitter into the target.
        # But the mask also gates UNnormalization at sampling time
        # (octo_model_pt.sample_actions: where(mask, a*std+mean, a)), so an
        # unmasked dim is emitted in raw radians with no rescaling -- the head
        # must then learn to output exactly ~0 by itself. Measured at step 2499
        # with the dims unmasked: predicted rotation std ~1.5 rad vs 1e-4 truth.
        # Normalized, the tiny std instead squashes any unlearned output to ~0.
        "action_normalization_mask": [True, True, True, True, True, True, False],
        # Data is already in octo-standard layout with delta-EEF actions;
        # no standardization/relabeling needed.
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

    # Small dataset (64 episodes / ~20.6k steps). Fewer steps than the 50k OXE default.
    max_steps = FieldReference(30000)
    window_size = FieldReference(default=2)  # octo-small-1.5 uses window_size=2

    config = dict(
        pretrained_path=placeholder(str),
        pretrained_step=placeholder(int),
        # single RTX 3060 (12 GB) -> small batch
        batch_size=16,
        shuffle_buffer_size=2000,
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
    workspace_augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
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
            "primary": (256, 256),
            "wrist": (128, 128),
        },
        image_augment_kwargs=dict(
            primary=workspace_augment_kwargs,
            wrist=wrist_augment_kwargs,
        ),
    )
    config["frame_transform_threads"] = 16
    config["traj_transform_kwargs"] = traj_transform_kwargs
    config["frame_transform_kwargs"] = frame_transform_kwargs
    return ConfigDict(config)
