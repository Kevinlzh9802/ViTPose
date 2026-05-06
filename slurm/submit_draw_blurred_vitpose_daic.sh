#!/bin/bash
#SBATCH --job-name=vitpose-blur-plot
#SBATCH --partition=insy,general
#SBATCH --qos=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --mail-type=END
#SBATCH --output=/home/nfs/zli33/slurm_outputs/vitpose-ingroup/blur_plot_slurm_%j.out
#SBATCH --error=/home/nfs/zli33/slurm_outputs/vitpose-ingroup/blur_plot_slurm_%j.err

set -euo pipefail

module use /opt/insy/modulefiles

NEON=/tudelft.net/staff-umbrella/neon
VITPOSE_DIR=/home/nfs/zli33/projects/ViTPose
SIF="${SIF:-${NEON}/apptainer/vitpose-0.0.5.sif}"

RESULTS_ROOT="${RESULTS_ROOT:-${NEON}/ingroup_dataset/B2_pipeline/vitpose_results}"
CAMERA_PARAMS_ROOT="${CAMERA_PARAMS_ROOT:-${NEON}/ingroup_dataset/processed_data/gopro_data/camera_calibration/camera_params}"
PLOT_DIR="${PLOT_DIR:-${NEON}/ingroup_dataset/B2_pipeline/person_plotting_blurred}"
FRAMES_ROOT="${FRAMES_ROOT:-${NEON}/ingroup_dataset/B2_pipeline/video_segs_raw}"
CAMERA_NUMBERS="${CAMERA_NUMBERS:-}"
PLOT_FRAME_INTERVAL="${PLOT_FRAME_INTERVAL:-1200}"
BLUR_RADIUS_PX="${BLUR_RADIUS_PX:-24}"
BLUR_KERNEL_SIZE="${BLUR_KERNEL_SIZE:-31}"

for arg in "$@"; do
    case "$arg" in
        --cam=*)
            CAMERA_NUMBERS="${arg#--cam=}"
            ;;
        --camera_numbers=*)
            CAMERA_NUMBERS="${arg#--camera_numbers=}"
            ;;
        --plot_dir=*)
            PLOT_DIR="${arg#--plot_dir=}"
            ;;
        --frames_root=*)
            FRAMES_ROOT="${arg#--frames_root=}"
            ;;
        --plot_frame_interval=*)
            PLOT_FRAME_INTERVAL="${arg#--plot_frame_interval=}"
            ;;
        --blur_radius_px=*)
            BLUR_RADIUS_PX="${arg#--blur_radius_px=}"
            ;;
        --blur_kernel_size=*)
            BLUR_KERNEL_SIZE="${arg#--blur_kernel_size=}"
            ;;
        *)
            echo "Error: unknown argument '$arg'" >&2
            echo "Usage: sbatch $0 [--cam=06,08,10] [--plot_dir=/path] [--plot_frame_interval=1200]" >&2
            exit 1
            ;;
    esac
done

if [[ ! -d "${RESULTS_ROOT}" ]]; then
    echo "Error: RESULTS_ROOT not found: ${RESULTS_ROOT}" >&2
    exit 1
fi

if [[ ! -d "${CAMERA_PARAMS_ROOT}" ]]; then
    echo "Error: CAMERA_PARAMS_ROOT not found: ${CAMERA_PARAMS_ROOT}" >&2
    exit 1
fi

mkdir -p "${PLOT_DIR}"

echo "Submitting blurred ViTPose keypoint drawing"
echo "  results_root=${RESULTS_ROOT}"
echo "  camera_params_root=${CAMERA_PARAMS_ROOT}"
echo "  plot_dir=${PLOT_DIR}"
echo "  frames_root=${FRAMES_ROOT}"
echo "  plot_frame_interval=${PLOT_FRAME_INTERVAL}"
echo "  blur_radius_px=${BLUR_RADIUS_PX}"
echo "  blur_kernel_size=${BLUR_KERNEL_SIZE}"
if [[ -n "${CAMERA_NUMBERS}" ]]; then
    echo "  camera_numbers=${CAMERA_NUMBERS}"
fi

python_args=(
    "${RESULTS_ROOT}"
    --camera_params_root "${CAMERA_PARAMS_ROOT}"
    --plot_dir "${PLOT_DIR}"
    --frames_root "${FRAMES_ROOT}"
    --plot_frame_interval "${PLOT_FRAME_INTERVAL}"
    --blur_radius_px "${BLUR_RADIUS_PX}"
    --blur_kernel_size "${BLUR_KERNEL_SIZE}"
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
    python ${VITPOSE_DIR}/demo/draw_blurred_vitpose_keypoints.py \
    "${python_args[@]}"

echo "Blurred keypoint figures written under ${PLOT_DIR}"

# ---------------------------------------------------------------------------
# Usage examples:
#
# 1) Run with defaults:
#    sbatch slurm/submit_draw_blurred_vitpose_daic.sh
#
# 2) Only draw specific cameras:
#    sbatch slurm/submit_draw_blurred_vitpose_daic.sh --cam=06,08,10
#
# 3) Adjust blur size:
#    BLUR_RADIUS_PX=32 BLUR_KERNEL_SIZE=41 \
#    sbatch slurm/submit_draw_blurred_vitpose_daic.sh
# ---------------------------------------------------------------------------
