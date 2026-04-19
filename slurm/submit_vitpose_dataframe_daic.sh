#!/bin/bash
#SBATCH --job-name=vitpose-df
#SBATCH --partition=insy,general
#SBATCH --qos=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
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
CAMERA_NUMBERS="${CAMERA_NUMBERS:-}"
GT_GROUPS_ROOT="${GT_GROUPS_ROOT:-${NEON}/ingroup_dataset/B2_pipeline/cgroup_annotation}"
PLOT_DIR="${PLOT_DIR:-${NEON}/ingroup_dataset/B2_pipeline/person_plotting}"
FRAMES_ROOT="${FRAMES_ROOT:-${NEON}/ingroup_dataset/B2_pipeline/video_segs_raw}"

for arg in "$@"; do
    case "$arg" in
        --cam=*)
            CAMERA_NUMBERS="${arg#--cam=}"
            ;;
        --camera_numbers=*)
            CAMERA_NUMBERS="${arg#--camera_numbers=}"
            ;;
        --gt_groups_root=*)
            GT_GROUPS_ROOT="${arg#--gt_groups_root=}"
            ;;
        --plot_dir=*)
            PLOT_DIR="${arg#--plot_dir=}"
            ;;
        --frames_root=*)
            FRAMES_ROOT="${arg#--frames_root=}"
            ;;
        *)
            echo "Error: unknown argument '$arg'" >&2
            echo "Usage: sbatch $0 [--cam=06,08,10]" >&2
            exit 1
            ;;
    esac
done

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
echo "  gt_groups_root=${GT_GROUPS_ROOT}"
echo "  plot_dir=${PLOT_DIR}"
echo "  frames_root=${FRAMES_ROOT}"
if [[ -n "${CAMERA_NUMBERS}" ]]; then
    echo "  camera_numbers=${CAMERA_NUMBERS}"
fi

python_args=(
    "${RESULTS_ROOT}"
    --output_dir "${OUTPUT_ROOT}"
    --output_name vitpose_dataframe.pkl
    --camera_params_root "${CAMERA_PARAMS_ROOT}"
    --gt_groups_root "${GT_GROUPS_ROOT}"
    --plot_dir "${PLOT_DIR}"
    --frames_root "${FRAMES_ROOT}"
)

if [[ -n "${CAMERA_NUMBERS}" ]]; then
    python_args+=(--camera_numbers "${CAMERA_NUMBERS}")
fi

apptainer exec \
    --containall \
    --env PYTHONPATH=${VITPOSE_DIR} \
    -B $HOME:$HOME \
    -B /tudelft.net/:/tudelft.net/ \
    ${SIF} \
    python ${VITPOSE_DIR}/demo/vitpose_to_dataframe.py \
    "${python_args[@]}"

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
# 
# 4) Only process specific cameras:
#    sbatch slurm/submit_vitpose_dataframe_daic.sh --cam=06,08,10
# ---------------------------------------------------------------------------
