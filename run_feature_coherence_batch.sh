#!/bin/bash
#SBATCH --job-name=viral_feature_coherence_batch
#SBATCH --output=logs/viral_feature_coherence_batch_%j.out
#SBATCH --error=logs/viral_feature_coherence_batch_%j.err
#SBATCH --time=04:00:00          # aumenta per run lunghi
#SBATCH --gpus=v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G         # un po' più generoso (COCO è pesante)
#SBATCH --nice=0
#SBATCH --mail-type=SUCCESS,FAIL         # ti avvisa se muore male
#SBATCH --mail-user=fbondi@ethz.ch  # opzionale

# -------------------------------
# ENV SETUP
# -------------------------------
echo "[INFO] Starting SLURM job $SLURM_JOB_ID on node $(hostname)"
echo "[INFO] Job started at $(date)"
echo "[INFO] Loading environment..."

# GPU sanity check
nvidia-smi || echo "[WARN] GPU not visible"

# Conda environment
source /cluster/project/cvg/students/fbondi/miniconda3/etc/profile.d/conda.sh
conda activate viral

module purge
module load eth_proxy || true

# -------------------------------
# CONFIG
# -------------------------------
export HF_HOME=/cluster/project/cvg/students/fbondi/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets

cd /cluster/project/cvg/students/fbondi/sem-project/VIRAL

# -------------------------------
# ANALYSIS COMMAND
# -------------------------------
echo "[INFO] Launching COCO feature coherence analysis..."
python analyse_feature_coherence_batch.py --start 50000 --end 60000

echo "[INFO] Job finished at $(date)"