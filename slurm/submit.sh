#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=35:59:00
#SBATCH --qos=medium
#SBATCH --gres=gpu:a40
#SBATCH --output=/home/nfs/zli33/slurm_outputs/vitpose-conflab/slurm_%j.out # Set name of output log. %j is the Slurm jobId
#SBATCH --error=/home/nfs/zli33/slurm_outputs/vitpose-conflab/slurm_%j.err # Set name of error log. %j is the Slurm jobId

# Set the CWD when the container is spin up
export APPTAINER_CWD=/workspace

# Run the container with:
# nvidia support
# set the PYTHONPATH for the local modules that are not installed properly via pip
# --bind $(pwd):/workspace to mount the code at the /workspace folder
# --bind /tmp:/tmp to avoid out of disk space errors inside the container on each evaluation step
apptainer run \
    --nv \
    --containall \
    --env PYTHONPATH=/workspace \
    --bind $(pwd):/workspace \
    --bind /tudelft.net/staff-umbrella/neon/experiments/VIT003/:/workspace/data/conflab \
    --bind /tmp:/tmp \
    /tudelft.net/staff-umbrella/neon/apptainer/vitpose-0.0.5.sif \
    python /workspace/tools/train.py configs/ViTPose_coco_plus_conflab_w_bg_256x192.py \
    --work-dir work_dirs/ViTPose_coco_plus_conflab_w_bg_256x192_filtered \
    --cfg-options model.pretrained=data/conflab/models/vitpose_base_coco_aic_mpii.pth --seed 0

# sbatch --job-name vitpose-conflab --account ewi-insy-prb --partition insy,general slurm/submit.sh
