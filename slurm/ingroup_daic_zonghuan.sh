#!/bin/bash
#SBATCH --job-name=vitpose-infer
#SBATCH --partition=insy,general
#SBATCH --qos=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:59:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 
#SBATCH --output=/home/nfs/zli33/slurm_outputs/vitpose-ingroup/infer_slurm_%j.out
#SBATCH --error=/home/nfs/zli33/slurm_outputs/vitpose-ingroup/infer_slurm_%j.err

module use /opt/insy/modulefiles
module load cuda/12.1

NEON=/tudelft.net/staff-umbrella/neon
VITPOSE_DIR=/home/nfs/zli33/projects/ViTPose

# ---------------------------------------------------------------------------
# Checkpoint — override by setting POSE_CKPT before sbatch, e.g.:
#   POSE_CKPT=${NEON}/code/ViTPose/work_dirs/.../best_AP_epoch_5.pth sbatch ...
# ---------------------------------------------------------------------------
POSE_CKPT="${POSE_CKPT:-${NEON}/code/ViTPose/work_dirs/ViTPose_coco_plus_conflab_w_bg_256x192_filtered/best_AP_epoch_1.pth}"

# ---------------------------------------------------------------------------
# Batch to run inference on — override by setting BATCH before sbatch, e.g.:
#   BATCH=228 sbatch slurm/submit_infer_daic.sh
# ---------------------------------------------------------------------------
BATCH="${BATCH:-228}"
DATA_ROOT="${DATA_ROOT:-${NEON}/ingroup_dataset/B2_pipeline/sam3_output/${BATCH}}"

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "Error: DATA_ROOT not found: ${DATA_ROOT}" >&2
    exit 1
fi

SIF="${SIF:-${NEON}/apptainer/vitpose-0.0.5.sif}"

# Output written directly to home directory
# OUT_DIR=/home/nfs/svacanas/outputs/vitpose/${BATCH}
OUT_DIR=${NEON}/ingroup_dataset/B2_pipeline/vitpose_results/${BATCH}
mkdir -p ${OUT_DIR}

# Unzip images to /tmp — zip contains flat jpgs (no images/ subfolder)
IMG_DIR=/tmp/ingroup_images/${BATCH}
mkdir -p ${IMG_DIR}

if [ -f "${DATA_ROOT}/images.zip" ]; then
    echo "Extracting images.zip to ${IMG_DIR}..."
    unzip -q "${DATA_ROOT}/images.zip" -d ${IMG_DIR}
    echo "Done extracting."
elif [ -d "${DATA_ROOT}/images" ]; then
    IMG_DIR=${DATA_ROOT}/images
else
    echo "Error: no images.zip or images/ found in ${DATA_ROOT}" >&2
    exit 1
fi

apptainer exec \
    --nv \
    --containall \
    --env PYTHONPATH=${VITPOSE_DIR} \
    -B $HOME:$HOME \
    -B /tudelft.net/:/tudelft.net/ \
    -B /tmp:/tmp \
    ${SIF} \
    python ${VITPOSE_DIR}/demo/infer_mask_bbox.py \
    ${VITPOSE_DIR}/configs/ViTPose_coco_plus_conflab_w_bg_256x192.py \
    ${POSE_CKPT} \
    --data-root ${DATA_ROOT} \
    --img-dir ${IMG_DIR} \
    --out-dir ${OUT_DIR} \
    --save-video

# Clean up /tmp
rm -rf /tmp/ingroup_images/${BATCH}

echo "Results saved to ${OUT_DIR}/vitpose_keypoints.json"

# ---------------------------------------------------------------------------
# Usage examples:
#
# 1) Default batch (cam06_batch01):
#    sbatch slurm/submit_infer_daic.sh
#
# 2) Specific batch:
#    BATCH=cam06_batch02 sbatch slurm/submit_infer_daic.sh
#
# 3) All batches:
#    for b in cam06_batch01 cam06_batch02 cam06_batch03 cam06_batch04 cam06_batch05 \
#             cam06_batch06 cam06_batch07 cam08_batch01 cam08_batch02 cam08_batch03 \
#             cam08_batch04 cam08_batch05 cam08_batch06 cam10_batch01 cam10_batch02 \
#             cam10_batch03; do BATCH=$b sbatch slurm/submit_infer_daic.sh; done
# ---------------------------------------------------------------------------