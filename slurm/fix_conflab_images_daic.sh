#!/bin/bash
#SBATCH --job-name=fix-conflab-images
#SBATCH --partition=insy,general
#SBATCH --qos=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0:30:00
#SBATCH --mail-type=END
#SBATCH --output=/home/nfs/zli33/slurm_outputs/vitpose-conflab/fix_images_%j.out
#SBATCH --error=/home/nfs/zli33/slurm_outputs/vitpose-conflab/fix_images_%j.err

set -euo pipefail

module use /opt/insy/modulefiles

NEON=/tudelft.net/staff-umbrella/neon
VITPOSE_DIR=/home/nfs/zli33/projects/ViTPose
SIF="${SIF:-${NEON}/apptainer/vitpose-0.0.5.sif}"

BBOX_KP_ROOT="${BBOX_KP_ROOT:-${NEON}/zonghuan/data/conflab/bbox_kp}"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
BATCHES_RAW="all"
DRY_RUN=""

for arg in "$@"; do
    case "$arg" in
        --batch=*)
            BATCHES_RAW="${arg#--batch=}"
            ;;
        --dry_run)
            DRY_RUN="--dry_run"
            ;;
        *)
            echo "Error: unknown argument '$arg'" >&2
            echo "Usage: sbatch $0 [--batch=228,431] [--dry_run]" >&2
            exit 1
            ;;
    esac
done

if [[ ! -d "${BBOX_KP_ROOT}" ]]; then
    echo "Error: BBOX_KP_ROOT not found: ${BBOX_KP_ROOT}" >&2
    exit 1
fi

echo "Conflab image layout fix"
echo "  bbox_kp_root=${BBOX_KP_ROOT}"
echo "  batches=${BATCHES_RAW}"
[[ -n "${DRY_RUN}" ]] && echo "  DRY RUN — no files will be moved"

apptainer exec \
    --containall \
    --env PYTHONPATH=${VITPOSE_DIR} \
    -B $HOME:$HOME \
    -B /tudelft.net/:/tudelft.net/ \
    ${SIF} \
    python ${VITPOSE_DIR}/demo/fix_conflab_images.py \
    "${BBOX_KP_ROOT}" \
    --batch "${BATCHES_RAW}" \
    ${DRY_RUN}

echo "Done."

# ---------------------------------------------------------------------------
# Notes:
#
# This is a one-time repair for batches where images.zip was extracted
# directly into bbox_kp/<batch>/ instead of bbox_kp/<batch>/images/.
# After running this script the layout will be:
#   bbox_kp/<batch>/images/00000100.jpg  (8-digit zero-padded frame id)
#
# Run a dry run first to verify:
#   sbatch slurm/fix_conflab_images_daic.sh --dry_run
#
# Then apply the fix:
#   sbatch slurm/fix_conflab_images_daic.sh
#
# Fix specific batches only:
#   sbatch slurm/fix_conflab_images_daic.sh --batch=228,431
#
# Override the bbox_kp root:
#   BBOX_KP_ROOT=/custom/path sbatch slurm/fix_conflab_images_daic.sh
# ---------------------------------------------------------------------------
