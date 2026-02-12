#!/bin/bash
#SBATCH --job-name="vitpose_conflab"
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=4000M
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=END
#SBATCH --account=research-eemcs-insy
#SBATCH --output=/home/zli33/slurm_outputs/vitpose_conflab/slurm_%j.out
#SBATCH --error=/home/zli33/slurm_outputs/vitpose_conflab/slurm_%j.err

home_path=/home/zli33
scratch_path=/scratch/zli33

model_path=$scratch_path/models/vitpose_conflab_filtered
data_path=$scratch_path/data/conflab/bbox_kp

# Subfolder under data_path (e.g. cam2); set DATA_SUBFOLDER when submitting
DATA_SUBFOLDER="${DATA_SUBFOLDER:-cam2}"
data_subfolder_path="$data_path/$DATA_SUBFOLDER"
if [[ ! -d "$data_subfolder_path" ]]; then
  echo "Error: subfolder not found: $data_subfolder_path" >&2
  exit 1
fi
# Path as seen inside the container (data_path is bound to /workspace/data)
DATA_ROOT="${DATA_ROOT:-/workspace/data/$DATA_SUBFOLDER}"

POSE_CKPT="${POSE_CKPT:-/workspace/models/best_AP_epoch_1.pth}"

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
