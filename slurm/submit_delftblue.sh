#!/bin/bash
#SBATCH --job-name="vitpose-conflab"
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4000M
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=END
#SBATCH --account=research-eemcs-insy
#SBATCH --output=/home/zli33/slurm_outputs/vitpose-conflab/slurm_%j.out
#SBATCH --error=/home/zli33/slurm_outputs/vitpose-conflab/slurm_%j.err

home_path=/home/zli33
scratch_path=/scratch/zli33

model_path=$scratch_path/models/vitpose_conflab_filtered
data_path=$scratch_path/data/sam4d/outputs/exp_20260208_220841_DWRI

POSE_CKPT="${POSE_CKPT:-/workspace/models/best_AP_epoch_1.pth}"
DATA_ROOT="${DATA_ROOT:-/workspace/data/masklets}"

export APPTAINER_CWD=/workspace
apptainer run \
    --nv \
    --containall \
    --env PYTHONPATH=/workspace \
    --bind $(pwd):/workspace \
    --bind $model_path:/workspace/models \
    --bind $data_path:/workspace/data \
    --bind /tmp:/tmp \
    $scratch_path/apptainers/vitpose-0.0.5.sif \
    python /workspace/demo/infer_mask_bbox.py \
    ${POSE_CKPT} \
    --data-root ${DATA_ROOT} \
    --save-video
