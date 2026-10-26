# GFlowTS

GFlowNet-based trajectory simplification with chunk-level candidate search,
frontier training, QCS-based fidelity evaluation, and global deletion repair.

## Installation

Use Python 3.10. For the CUDA 12.1 environment:

```bash
python -m pip install -r requirements.txt
```

## Data format

Place trajectories in a directory `data/trajectories/`. Each trajectory
is a text file named with its integer index, for example `0`, `1`. Each line contains three space-separated numeric columns:

```text
x y timestamp
```

## Training

```bash
python train_streaming.py --traj_path data/trajectories --start_idx 0 --end_idx 1000 --val_start_idx 1000 --val_end_idx 1500 --groups 0 --epochs 50 --f1_threshold 0.85 --model_path checkpoints/model.pt --use_gpu
```

## Verification

```bash
python verify_streaming.py --traj_path data/trajectories --checkpoint checkpoints/model_epoch50.pt --start_idx 1500 --end_idx 2000 --target_f1_list 0.85 --summary_csv results/summary.csv --per_traj_csv_dir results/per_trajectory
```

## Repository contents

- `train_streaming.py`: training and shared candidate inference.
- `verify_streaming.py`: checkpoint verification and metric output.
- `gflownet/`: model, trainer, environment, frontier, repair, and QCS utilities.
- `data_utils.py`: trajectory loading and RDP preprocessing.
- `profiling_utils.py`: optional runtime measurement.
