"""Finetune config for UR10e pick-cup WITH proprioceptive state input.

Same as finetune_ur10e.py, but feeds robot proprioception to the model. Two changes:

  1. proprio_obs_key = "joint_state"  (the 6 UR joint angles).
     We use joint_state, NOT the EEF `state` field: `state`'s roll channel wraps
     across +-pi (std ~3.0, 2*pi step-jumps), which becomes noise once fed to the
     model. joint_state is smooth on all 6 dims. See the data check in the eval notes.

  2. update_config adds a `proprio` observation tokenizer. octo-small-1.5 ships with
     ONLY primary/wrist image tokenizers and a language task tokenizer -- it has NO
     proprio tokenizer, so setting proprio_obs_key alone would load the data but the
     model would never consume it. We inject a LowdimObsTokenizer (JAX module path;
     finetune_pt.py's _jax_config_to_pt_config appends _pt/Pt) and extend
     num_tokens_dict with proprio=6. The new obs_proprio projection + positional
     embedding have no counterpart in the pretrained checkpoint, so they train from
     scratch -- fine in `full` mode. Everything else is loaded from the checkpoint.

Usage:
  torchrun --nproc_per_node 1 scripts/finetune_pt.py \
    --config scripts/configs/finetune_ur10e_proprio.py:full,language_conditioned \
    --config.pretrained_path=hf://rail-berkeley/octo-small-1.5
"""
from ml_collections import ConfigDict
from ml_collections.config_dict import FieldReference, placeholder

from octo.utils.spec import ModuleSpec

# Absolute path to the RLDS output dir written by convert_ur_to_rlds.py --out rlds
DATA_DIR = "/home/khanh/work/octo-pytorch/rlds"

# UR joint_state is 6-dim -> 6 proprio tokens. The other counts are octo-small-1.5's
# fixed token budget (primary 16x16=256, wrist 8x8=64, T5 max_length 16, 1 action
# readout); num_tokens_dict has no default 'proprio' key, so we must list them all.
NUM_TOKENS_DICT = {"primary": 256, "wrist": 64, "language": 16, "action": 1, "proprio": 6}


def get_config(config_string="full,language_conditioned"):
    mode, task = config_string.split(",")
    assert task in ["image_conditioned", "language_conditioned", "multimodal"]
    assert mode in ["full", "head_only", "head_mlp_only"]

    FINETUNING_KWARGS = {
        "name": "ur10e_pick_cup",
        "data_dir": DATA_DIR,
        "image_obs_keys": {"primary": "image", "wrist": "wrist_image"},
        # Feed the 6 UR joint angles as proprioception. The pipeline renames this
        # observation key to "proprio" and normalizes it to mean 0 / std 1.
        "proprio_obs_key": "joint_state",
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
    window_size = FieldReference(default=2)  # octo-small-1.5 uses window_size=2

    config = dict(
        pretrained_path=placeholder(str),
        pretrained_step=placeholder(int),
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
        # Injected into the pretrained model config by finetune_pt.py
        # (config.update(update_config)) BEFORE the jax->pt conversion, so the
        # tokenizer spec uses the JAX module path.
        update_config=dict(
            model=dict(
                observation_tokenizers=dict(
                    proprio=ModuleSpec.create(
                        "octo.model.components.tokenizers:LowdimObsTokenizer",
                        obs_keys=["proprio"],
                        discretize=False,
                    ),
                ),
                num_tokens_dict=NUM_TOKENS_DICT,
            ),
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
