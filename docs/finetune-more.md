# FINETUNE v1.1

## v1.1.0

stride=1

Finetune
```bash
cd /home/khanh/work/octo-pytorch && \
screen -dmS octo_ur10e_v11 bash -ic '
  CONFIG=scripts/configs/finetune_ur10e.py \
  NAME=ur10e_pick_cup_v11 \
  SAVE_DIR=/mnt/data/ur10e-robotiq/octo-small-finetune-v11/checkpoints \
  LOG=/mnt/data/ur10e-robotiq/octo-small-finetune-v11/finetune_ur10e_v11.log \
  NUM_STEPS=60000 \
  ./scripts/run_finetune_ur10e.sh \
    --config.dataset_kwargs.name=ur10e_pick_cup:1.1.0 \
    --config.shuffle_buffer_size=1000'
```

Monitor
```bash
screen -r octo_ur10e_v11                 # Ctrl-A D to detach
tail -f /mnt/data/ur10e-robotiq/octo-small-finetune-v11/finetune_ur10e_v11.log | grep train_loss
watch -n30 'free -h | grep -i swap'      # swap should stay well under 16G
```

## v1.1.1

stride=1, camera=wrist-only

Finetune
```bash
cd /home/khanh/work/octo-pytorch && \
screen -dmS octo_ur10e_v111 bash -ic '
  CONFIG=scripts/configs/finetune_ur10e_wrist.py \
  NAME=ur10e_pick_cup_v111 \
  SAVE_DIR=/mnt/data/ur10e-robotiq/octo-small-finetune-v111/checkpoints \
  LOG=/mnt/data/ur10e-robotiq/octo-small-finetune-v111/finetune_ur10e_v111.log \
  NUM_STEPS=60000 \
  ./scripts/run_finetune_ur10e.sh --config.dataset_kwargs.name=ur10e_pick_cup:1.1.0'
```

Monitor
```bash
screen -r octo_ur10e_v111        # Ctrl-A D to detach
tail -f /mnt/data/ur10e-robotiq/octo-small-finetune-v111/finetune_ur10e_v111.log | grep train_loss
```

## v1.1.2

stride=2

Finetune
```bash
cd /home/khanh/work/octo-pytorch && \
screen -dmS octo_ur10e_v112 bash -ic '
  CONFIG=scripts/configs/finetune_ur10e.py \
  NAME=ur10e_pick_cup_v112 \
  SAVE_DIR=/mnt/data/ur10e-robotiq/octo-small-finetune-v112/checkpoints \
  LOG=/mnt/data/ur10e-robotiq/octo-small-finetune-v112/finetune_ur10e_v112.log \
  NUM_STEPS=60000 \
  ./scripts/run_finetune_ur10e.sh \
    --config.dataset_kwargs.name=ur10e_pick_cup:1.1.2 \
    --config.shuffle_buffer_size=1000'
```

Monitor
```bash
screen -r octo_ur10e_v112                # Ctrl-A D to detach
tail -f /mnt/data/ur10e-robotiq/octo-small-finetune-v112/finetune_ur10e_v112.log | grep train_loss
watch -n30 'free -h | grep -i swap'      # swap should stay well under 16G
```
