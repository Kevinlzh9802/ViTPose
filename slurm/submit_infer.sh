#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=11:59:00
#SBATCH --qos=short
#SBATCH --gres=gpu:a40
#SBATCH --job-name=vitpose-infer
#SBATCH --account ewi-insy-prb
#SBATCH --partition insy,general
#SBATCH --output=/home/nfs/zli33/slurm_outputs/vitpose-conflab/infer_slurm_%j.out
#SBATCH --error=/home/nfs/zli33/slurm_outputs/vitpose-conflab/infer_slurm_%j.err

# Default checkpoint -- override by setting POSE_CKPT before sbatch, e.g.:
#   POSE_CKPT=work_dirs/.../best_AP_epoch_5.pth sbatch slurm/submit_infer.sh
# use virtual path in apptainer
POSE_CKPT="${POSE_CKPT:-data/conflab/models/vitpose_base_coco_aic_mpii.pth}"

# Data root to infer on -- override by setting DATA_ROOT before sbatch, e.g.:
#   DATA_ROOT=data/conflab/my_segment sbatch slurm/submit_infer.sh
DATA_ROOT="${DATA_ROOT:-data/conflab}"

# Set the CWD when the container is spin up
export APPTAINER_CWD=/workspace

apptainer run \
    --nv \
    --containall \
    --env PYTHONPATH=/workspace \
    --bind $(pwd):/workspace \
    --bind /tudelft.net/staff-umbrella/neon/experiments/VIT003/:/workspace/data/conflab \
    --bind /tmp:/tmp \
    /tudelft.net/staff-umbrella/neon/apptainer/vitpose-0.0.5.sif \
    python /workspace/demo/infer_mask_bbox.py \
    ${POSE_CKPT} \
    --data-root ${DATA_ROOT} \
    --save-video

# Usage examples:
# 1) With defaults (pretrained checkpoint):
#    sbatch slurm/submit_infer.sh
#
# 2) With a fine-tuned checkpoint:
#    POSE_CKPT=work_dirs/ViTPose_coco_plus_conflab_w_bg_256x192_filtered/best_AP_epoch_5.pth \
#    DATA_ROOT=data/conflab/my_segment \
#    sbatch slurm/submit_infer.sh
