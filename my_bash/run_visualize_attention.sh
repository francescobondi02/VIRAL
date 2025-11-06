#!/bin/bash
#SBATCH --job-name=visualize_attn
#SBATCH --output=logs/visualize_attn_%j.out
#SBATCH --error=logs/visualize_attn_%j.err
#SBATCH --time=00:20:00
#SBATCH --gpus=v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G

# -------------------------------
# ENV SETUP
# -------------------------------
echo "[INFO] Starting SLURM job $SLURM_JOB_ID on node $(hostname)"

# Attiva Conda
source /cluster/project/cvg/students/fbondi/miniconda3/etc/profile.d/conda.sh
conda activate viral

module purge
module load eth_proxy || true

nvidia-smi

# -------------------------------
# HUGGINGFACE CACHE CONFIG
# -------------------------------
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/cluster/project/cvg/students/fbondi/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets

# -------------------------------
# EVALUATION COMMAND
# -------------------------------
echo "[INFO] Launching Visualize Attentions..."
python my_scripts/visualize_attention.py
echo "[INFO] Job finished at $(date)"