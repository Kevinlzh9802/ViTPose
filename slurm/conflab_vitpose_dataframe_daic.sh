#!/bin/bash
#SBATCH --job-name=vitpose-df-conflab
#SBATCH --partition=insy,general
#SBATCH --qos=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --mail-type=END
#SBATCH --output=/home/nfs/zli33/slurm_outputs/vitpose-conflab/df_slurm_%j.out
#SBATCH --error=/home/nfs/zli33/slurm_outputs/vitpose-conflab/df_slurm_%j.err

set -euo pipefail

module use /opt/insy/modulefiles

NEON=/tudelft.net/staff-umbrella/neon
VITPOSE_DIR=/home/nfs/zli33/projects/ViTPose
SIF="${SIF:-${NEON}/apptainer/vitpose-0.0.5.sif}"

# ---------------------------------------------------------------------------
# Camera mapping: first digit of each batch number → camera digit X
#   intrinsic: /neon/zonghuan/data/conflab/intrinsics/intrinsic_X.json
#   extrinsic: EXTRINSICS_DIR/extrinsic_X_zh.json
#
# Note: conflab extrinsics are in centimetres, so body_height is passed as
# 170 (cm) instead of the default 1.7 (m) used for the ingroup dataset.
# ---------------------------------------------------------------------------

CONFLAB_INTRINSICS_DIR="${NEON}/zonghuan/data/conflab/intrinsics"
EXTRINSICS_DIR="${EXTRINSICS_DIR:-${NEON}/zonghuan/data/conflab/extrinsics}"
BODY_HEIGHT=170

VITPOSE_OUTPUTS="${NEON}/zonghuan/data/conflab/vitpose_outputs"
OUTPUT_ROOT="${OUTPUT_ROOT:-${NEON}/zonghuan/data/conflab/vitpose_dataframe}"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
BATCHES_RAW=""
PLOT_DIR=""
FRAMES_ROOT=""

for arg in "$@"; do
    case "$arg" in
        --batch=*)
            BATCHES_RAW="${arg#--batch=}"
            ;;
        --plot_dir=*)
            PLOT_DIR="${arg#--plot_dir=}"
            ;;
        --frames_root=*)
            FRAMES_ROOT="${arg#--frames_root=}"
            ;;
        *)
            echo "Error: unknown argument '$arg'" >&2
            echo "Usage: sbatch $0 --batch=228,229,431 [--plot_dir=/path] [--frames_root=/path]" >&2
            exit 1
            ;;
    esac
done

# Fall back to BATCH env var for single-batch backwards compatibility.
if [[ -z "${BATCHES_RAW}" ]]; then
    if [[ -z "${BATCH:-}" ]]; then
        echo "Error: specify batches via --batch=228,229 or the BATCH env var" >&2
        exit 1
    fi
    BATCHES_RAW="${BATCH}"
fi

# ---------------------------------------------------------------------------
# Build a single temp directory containing one subfolder per batch and one
# camera_params entry per distinct camera digit.  vitpose_to_dataframe.py is
# then invoked once across all batches.
#
#   TEMP_DIR/
#     cam<NN>_batch<BBB>/
#       vitpose_keypoints.json  → vitpose_outputs/<BBB>/vitpose_keypoints.json
#     camera_params/
#       camera_<NN>/
#         intrinsic.json        → intrinsics/intrinsic_<X>.json
#         extrinsic.json        → extrinsics/extrinsic_<X>_zh.json
# ---------------------------------------------------------------------------
TEMP_DIR=$(mktemp -d /tmp/vitpose_df_XXXXXX)
cleanup() { rm -rf "${TEMP_DIR}"; }
trap cleanup EXIT

mkdir -p "${OUTPUT_ROOT}"

echo "ViTPose dataframe conversion (conflab)"
echo "  batches=${BATCHES_RAW}"
echo "  output_root=${OUTPUT_ROOT}"

IFS=',' read -ra BATCH_LIST <<< "${BATCHES_RAW}"

for BATCH in "${BATCH_LIST[@]}"; do
    BATCH="${BATCH// /}"   # trim any spaces around commas
    CAM_DIGIT="${BATCH:0:1}"
    CAM_NUMBER=$(printf "%02d" "${CAM_DIGIT}")

    SOURCE_JSON="${VITPOSE_OUTPUTS}/${BATCH}/vitpose_keypoints.json"
    INTRINSIC_FILE="${CONFLAB_INTRINSICS_DIR}/intrinsic_${CAM_DIGIT}.json"
    EXTRINSIC_FILE="${EXTRINSICS_DIR}/extrinsic_${CAM_DIGIT}_zh.json"

    if [[ ! -f "${SOURCE_JSON}" ]]; then
        echo "Error: source JSON not found: ${SOURCE_JSON}" >&2; exit 1
    fi
    if [[ ! -f "${INTRINSIC_FILE}" ]]; then
        echo "Error: intrinsic file not found: ${INTRINSIC_FILE}" >&2; exit 1
    fi
    if [[ ! -f "${EXTRINSIC_FILE}" ]]; then
        echo "Error: extrinsic file not found: ${EXTRINSIC_FILE}" >&2; exit 1
    fi

    # Symlink the keypoints JSON under a cam-named folder.
    CAM_BATCH_DIR="${TEMP_DIR}/cam${CAM_NUMBER}_batch${BATCH}"
    mkdir "${CAM_BATCH_DIR}"
    ln -s "${SOURCE_JSON}" "${CAM_BATCH_DIR}/vitpose_keypoints.json"
    echo "  batch=${BATCH} → cam${CAM_NUMBER}  (${SOURCE_JSON})"

    # Symlink camera params once per distinct camera digit.
    TEMP_CAM_PARAMS="${TEMP_DIR}/camera_params/camera_${CAM_NUMBER}"
    if [[ ! -d "${TEMP_CAM_PARAMS}" ]]; then
        mkdir -p "${TEMP_CAM_PARAMS}"
        ln -s "${INTRINSIC_FILE}" "${TEMP_CAM_PARAMS}/intrinsic.json"
        ln -s "${EXTRINSIC_FILE}" "${TEMP_CAM_PARAMS}/extrinsic.json"
    fi
done

python_args=(
    "${TEMP_DIR}"
    --output_dir "${OUTPUT_ROOT}"
    --output_name vitpose_dataframe.pkl
    --camera_params_root "${TEMP_DIR}/camera_params"
    --body_height "${BODY_HEIGHT}"
)

if [[ -n "${PLOT_DIR}" ]]; then
    python_args+=(--plot_dir "${PLOT_DIR}")
fi
if [[ -n "${FRAMES_ROOT}" ]]; then
    python_args+=(--frames_root "${FRAMES_ROOT}")
fi

apptainer exec \
    --containall \
    --env PYTHONPATH=${VITPOSE_DIR} \
    -B $HOME:$HOME \
    -B /tudelft.net/:/tudelft.net/ \
    -B /tmp:/tmp \
    ${SIF} \
    python ${VITPOSE_DIR}/demo/vitpose_to_dataframe.py \
    "${python_args[@]}"

echo "Dataframes written under ${OUTPUT_ROOT}"

# ---------------------------------------------------------------------------
# Notes:
#
# 1) intrinsic_X.json must contain "model": "pinhole".
#    The loader defaults to "fisheye" when the field is absent. With 5
#    distortion coefficients and no model field it silently truncates to the
#    first 4 and calls cv2.fisheye.undistortPoints — producing wrong
#    back-projection. Add "model": "pinhole" to every intrinsic file.
#
# 2) Output world coordinates (spaceFeat x/y) are in centimetres.
#    Conflab extrinsics use centimetres, so body_height is set to 170 (cm)
#    to match. Multiply x/y by 0.01 downstream if metres are expected.
#
# ---------------------------------------------------------------------------
# Usage examples:
#
# 1) Single batch:
#    sbatch slurm/conflab_vitpose_dataframe_daic.sh --batch=228
#
# 2) Multiple batches in one job:
#    sbatch slurm/conflab_vitpose_dataframe_daic.sh --batch=228,229,431
#
# 3) With diagnostic plots:
#    sbatch slurm/conflab_vitpose_dataframe_daic.sh --batch=228,229 --plot_dir=/path/to/plots
#
# 4) Override output root or extrinsics location:
#    OUTPUT_ROOT=/custom/output sbatch slurm/conflab_vitpose_dataframe_daic.sh --batch=228
#    EXTRINSICS_DIR=/alt/extrinsics sbatch slurm/conflab_vitpose_dataframe_daic.sh --batch=228
#
# 5) Backwards-compatible single-batch via env var:
#    BATCH=228 sbatch slurm/conflab_vitpose_dataframe_daic.sh
# ---------------------------------------------------------------------------
