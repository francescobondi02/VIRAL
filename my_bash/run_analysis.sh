#!/bin/bash
#SBATCH --job-name=viral_coco_analysis
#SBATCH --output=logs/viral_coco_analysis_%j.out
#SBATCH --error=logs/viral_coco_analysis_%j.err
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
export HF_HOME=/cluster/project/cvg/students/fbondi/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets

# -------------------------------
# PATH CONFIGURATION
# -------------------------------
MODEL_PATH="liuhaotian/llava-v1.5-7b-lora"

# Usa un’immagine specifica del dataset COCO
COCO_IMG_DIR="/cluster/project/cvg/data/mscoco/mscoco/train2017"
IMAGE_PATH="${COCO_IMG_DIR}/000000000009.jpg"   # Cambiala se vuoi testare un’altra

OUTPUT_DIR="./outputs/features"
OUTPUT_PATH="${OUTPUT_DIR}/coco_sample.pt"

# -------------------------------
# ANALYSIS COMMAND
# -------------------------------
echo "[INFO] Launching COCO feature analysis..."
mkdir -p "$OUTPUT_DIR"

python analyse_visual_tokens.py \
  --model-path "$MODEL_PATH" \
  --image-path "$IMAGE_PATH" \
  --output-path "$OUTPUT_PATH" \
  --fp16

echo "[INFO] Saved output to $OUTPUT_PATH"
echo "[INFO] Job finished at $(date)"