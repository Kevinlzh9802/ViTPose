#!/bin/bash
#SBATCH --job-name=vitpose-df
#SBATCH --partition=insy,general
#SBATCH --qos=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:59:00
#SBATCH --mail-type=END     # Set mail type to 'END' to receive a mail when the job finishes. 
#SBATCH --output=/home/nfs/zli33/slurm_outputs/vitpose-ingroup/infer_slurm_%j.out
#SBATCH --error=/home/nfs/zli33/slurm_outputs/vitpose-ingroup/infer_slurm_%j.err

set -euo pipefail

module use /opt/insy/modulefiles

NEON=/tudelft.net/staff-umbrella/neon
VITPOSE_DIR=/home/nfs/zli33/projects/ViTPose
SIF="${SIF:-${NEON}/apptainer/vitpose-0.0.5.sif}"

RESULTS_ROOT="${RESULTS_ROOT:-${NEON}/ingroup_dataset/B2_pipeline/vitpose_results}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${NEON}/ingroup_dataset/B2_pipeline/vitpose_dataframe}"

CAMERA_PARAMS_ROOT="${CAMERA_PARAMS_ROOT:-${NEON}/ingroup_dataset/processed_data/gopro_data/camera_calibration/camera_params}"

if [[ ! -d "${RESULTS_ROOT}" ]]; then
    echo "Error: RESULTS_ROOT not found: ${RESULTS_ROOT}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

if [[ ! -d "${CAMERA_PARAMS_ROOT}" ]]; then
    echo "Warning: CAMERA_PARAMS_ROOT not found: ${CAMERA_PARAMS_ROOT}" >&2
fi

echo "Submitting ViTPose dataframe conversion"
echo "  results_root=${RESULTS_ROOT}"
echo "  output_root=${OUTPUT_ROOT}"
echo "  camera_params_root=${CAMERA_PARAMS_ROOT}"

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
    --camera_params_root "${CAMERA_PARAMS_ROOT}"

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
# 3) Override camera params root:
#    CAMERA_PARAMS_ROOT=/path/to/camera_params \
#    sbatch slurm/submit_vitpose_dataframe_daic.sh
# ---------------------------------------------------------------------------
