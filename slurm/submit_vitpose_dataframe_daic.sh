#!/bin/bash
#SBATCH --job-name=vitpose-df
#SBATCH --partition=insy,general
#SBATCH --qos=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:59:00
#SBATCH --output=vitpose_dataframe_%j.out
#SBATCH --error=vitpose_dataframe_%j.err

set -euo pipefail

module use /opt/insy/modulefiles

NEON=/tudelft.net/staff-umbrella/neon
VITPOSE_DIR=/home/nfs/zli33/projects/ViTPose
SIF="${SIF:-${NEON}/apptainer/vitpose-0.0.5.sif}"

RESULTS_ROOT="${RESULTS_ROOT:-${NEON}/ingroup_dataset/B2_pipeline/vitpose_results}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${NEON}/ingroup_dataset/B2_pipeline/vitpose_dataframe}"

CALIB_ROOT="${CALIB_ROOT:-${NEON}/ingroup_dataset/processed_data/gopro_data/camera_calibration}"
INTRINSICS_DIR="${INTRINSICS_DIR:-${CALIB_ROOT}/intrinsics}"
EXTRINSICS_DIR="${EXTRINSICS_DIR:-${CALIB_ROOT}/extrinsics}"

if [[ ! -d "${RESULTS_ROOT}" ]]; then
    echo "Error: RESULTS_ROOT not found: ${RESULTS_ROOT}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

if [[ ! -d "${INTRINSICS_DIR}" ]]; then
    echo "Warning: INTRINSICS_DIR not found: ${INTRINSICS_DIR}" >&2
fi

if [[ ! -d "${EXTRINSICS_DIR}" ]]; then
    echo "Warning: EXTRINSICS_DIR not found: ${EXTRINSICS_DIR}" >&2
fi

echo "Submitting ViTPose dataframe conversion"
echo "  results_root=${RESULTS_ROOT}"
echo "  output_root=${OUTPUT_ROOT}"
echo "  intrinsics_dir=${INTRINSICS_DIR}"
echo "  extrinsics_dir=${EXTRINSICS_DIR}"

apptainer exec \
    --containall \
    --env PYTHONPATH=${VITPOSE_DIR} \
    -B $HOME:$HOME \
    -B /tudelft.net/:/tudelft.net/ \
    ${SIF} \
    python ${VITPOSE_DIR}/demo/vitpose_to_dataframe.py \
    "${RESULTS_ROOT}" \
    --output_dir "${OUTPUT_ROOT}" \
    --output_name vitpose_dataframe.pkl \
    --intrinsics_dir "${INTRINSICS_DIR}" \
    --extrinsics_dir "${EXTRINSICS_DIR}"

echo "Dataframes written under ${OUTPUT_ROOT}"

# ---------------------------------------------------------------------------
# Usage examples:
#
# 1) Run with defaults:
#    sbatch slurm/submit_vitpose_dataframe_daic.sh
#
# 2) Override input/output roots:
#    RESULTS_ROOT=/path/to/vitpose_results \
#    OUTPUT_ROOT=/path/to/B2_pipeline/vitpose_dataframe \
#    sbatch slurm/submit_vitpose_dataframe_daic.sh
#
# 3) Override calibration directories:
#    INTRINSICS_DIR=/path/to/intrinsics \
#    EXTRINSICS_DIR=/path/to/extrinsics \
#    sbatch slurm/submit_vitpose_dataframe_daic.sh
# ---------------------------------------------------------------------------
