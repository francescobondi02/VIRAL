#!/bin/bash
#SBATCH --job-name=viral_feature_coherence
#SBATCH --output=logs/viral_feature_coherence_%j.out
#SBATCH --error=logs/viral_feature_coherence_%j.err
#SBATCH --time=00:05:00
#SBATCH --gpus=nvidia_geforce_rtx_2080_ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G

# -------------------------------
# ENV SETUP
# -------------------------------
echo "[INFO] Starting SLURM job $SLURM_JOB_ID on node $(hostname)"

nvidia-smi

# Attiva Conda
source /cluster/project/cvg/students/fbondi/miniconda3/etc/profile.d/conda.sh
conda activate viral

module purge
module load eth_proxy || true

# -------------------------------
# ANALYSIS COMMAND
# -------------------------------
echo "[INFO] Launching COCO feature coherence..."

python analyse_feature_coherence.py
echo "[INFO] Job finished at $(date)"