#!/bin/bash
#SBATCH --job-name=vitpose-plot-conflab
#SBATCH --partition=insy,general
#SBATCH --qos=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --mail-type=END
#SBATCH --output=/home/nfs/zli33/slurm_outputs/vitpose-conflab/plot_slurm_%j.out
#SBATCH --error=/home/nfs/zli33/slurm_outputs/vitpose-conflab/plot_slurm_%j.err

set -euo pipefail

module use /opt/insy/modulefiles

NEON=/tudelft.net/staff-umbrella/neon
VITPOSE_DIR=/home/nfs/zli33/projects/ViTPose
SIF="${SIF:-${NEON}/apptainer/vitpose-0.0.5.sif}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${NEON}/zonghuan/data/conflab/vitpose_dataframe}"
FRAME_INTERVAL="${FRAME_INTERVAL:-300}"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
BATCHES_RAW=""

for arg in "$@"; do
    case "$arg" in
        --batch=*)
            BATCHES_RAW="${arg#--batch=}"
            ;;
        --frame_interval=*)
            FRAME_INTERVAL="${arg#--frame_interval=}"
            ;;
        *)
            echo "Error: unknown argument '$arg'" >&2
            echo "Usage: sbatch $0 --batch=228,229,431 [--frame_interval=300]" >&2
            exit 1
            ;;
    esac
done

# Fall back to BATCH env var for single-batch backwards compatibility.
if [[ -z "${BATCHES_RAW}" ]]; then
    if [[ -z "${BATCH:-}" ]]; then
        echo "Error: specify batches via --batch=228,229, --batch=all, or the BATCH env var" >&2
        exit 1
    fi
    BATCHES_RAW="${BATCH}"
fi

if [[ ! -d "${OUTPUT_ROOT}" ]]; then
    echo "Error: OUTPUT_ROOT not found: ${OUTPUT_ROOT}" >&2
    exit 1
fi

echo "Conflab BEV position plots"
echo "  output_root=${OUTPUT_ROOT}"
echo "  batches=${BATCHES_RAW}"
echo "  frame_interval=${FRAME_INTERVAL}"

apptainer exec \
    --containall \
    --env PYTHONPATH=${VITPOSE_DIR} \
    -B $HOME:$HOME \
    -B /tudelft.net/:/tudelft.net/ \
    ${SIF} \
    python ${VITPOSE_DIR}/demo/plot_conflab_dataframes.py \
    "${OUTPUT_ROOT}" \
    --batch "${BATCHES_RAW}" \
    --frame_interval "${FRAME_INTERVAL}"

echo "Plots written under ${OUTPUT_ROOT}"

# ---------------------------------------------------------------------------
# Notes:
#
# 1) Plots are stored directly in each batch folder alongside the pkl:
#    OUTPUT_ROOT/cam<NN>_batch<BBB>/<batch_name>__frame_<id>.png
#
# 2) Conflab world coordinates are in centimetres. The Python script scales
#    x/y by 0.01 before plotting so axes are in metres and circle/arrow
#    sizes in plot_person.py render at the correct physical scale.
#
# ---------------------------------------------------------------------------
# Usage examples:
#
# 1) Single batch:
#    sbatch slurm/conflab_plot_daic.sh --batch=228
#
# 2) Multiple batches:
#    sbatch slurm/conflab_plot_daic.sh --batch=228,229,431
#
# 3) All available batches:
#    sbatch slurm/conflab_plot_daic.sh --batch=all
#
# 4) Custom frame interval (e.g. every 10 s):
#    sbatch slurm/conflab_plot_daic.sh --batch=228 --frame_interval=600
#
# 5) Override output root:
#    OUTPUT_ROOT=/custom/output sbatch slurm/conflab_plot_daic.sh --batch=all
#
# 6) Backwards-compatible single-batch via env var:
#    BATCH=228 sbatch slurm/conflab_plot_daic.sh
# ---------------------------------------------------------------------------
