#!/bin/bash
#SBATCH --job-name=vitpose-to-gcff
#SBATCH --partition=insy,general
#SBATCH --qos=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --mail-type=END
#SBATCH --output=/home/nfs/zli33/slurm_outputs/vitpose-conflab/to_gcff_%j.out
#SBATCH --error=/home/nfs/zli33/slurm_outputs/vitpose-conflab/to_gcff_%j.err

set -euo pipefail

module use /opt/insy/modulefiles

NEON=/tudelft.net/staff-umbrella/neon
VITPOSE_DIR=/home/nfs/zli33/projects/ViTPose
SIF="${SIF:-${NEON}/apptainer/vitpose-0.0.5.sif}"

VITPOSE_ROOT="${VITPOSE_ROOT:-${NEON}/zonghuan/data/conflab/vitpose_dataframe}"
OUTPUT="${OUTPUT:-${NEON}/zonghuan/data/conflab/GCFF/data.pkl}"
WORLD_SCALE="${WORLD_SCALE:-0.01}"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
BATCHES_RAW="all"

for arg in "$@"; do
    case "$arg" in
        --batch=*)
            BATCHES_RAW="${arg#--batch=}"
            ;;
        --output=*)
            OUTPUT="${arg#--output=}"
            ;;
        --world_scale=*)
            WORLD_SCALE="${arg#--world_scale=}"
            ;;
        *)
            echo "Error: unknown argument '$arg'" >&2
            echo "Usage: sbatch $0 [--batch=228,229] [--output=/path/data.pkl] [--world_scale=0.01]" >&2
            exit 1
            ;;
    esac
done

echo "ViTPose → GCFF converter"
echo "  vitpose_root=${VITPOSE_ROOT}"
echo "  output=${OUTPUT}"
echo "  batches=${BATCHES_RAW}"
echo "  world_scale=${WORLD_SCALE}"

apptainer exec \
    --containall \
    --env PYTHONPATH=${VITPOSE_DIR} \
    -B $HOME:$HOME \
    -B /tudelft.net/:/tudelft.net/ \
    ${SIF} \
    python ${VITPOSE_DIR}/demo/vitpose_to_gcff.py \
    --vitpose_root "${VITPOSE_ROOT}" \
    --output "${OUTPUT}" \
    --batch "${BATCHES_RAW}" \
    --world_scale "${WORLD_SCALE}"

echo "Saved: ${OUTPUT}"

# ---------------------------------------------------------------------------
# Notes:
#
# Output: data.pkl — merged GCFF-schema DataFrame, no GCFF detections.
# Then run GCFF on it:
#   sbatch FF_conflab/slurm/submit_gcff_vitpose.sh --mode=gcff
#
# world_scale=0.01 converts ViTPose cm coords → metres (GCFF default param scale).
# If GCFF params were tuned on cm-scale data, pass --world_scale=1.0.
#
# Usage examples:
#
# 1) All available batches (default):
#    sbatch slurm/vitpose_to_gcff_daic.sh
#
# 2) Specific batches:
#    sbatch slurm/vitpose_to_gcff_daic.sh --batch=228,229,431
#
# 3) Keep cm units:
#    sbatch slurm/vitpose_to_gcff_daic.sh --world_scale=1.0
#
# 4) Override paths:
#    OUTPUT=/custom/path/data.pkl sbatch slurm/vitpose_to_gcff_daic.sh
# ---------------------------------------------------------------------------
