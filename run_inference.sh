#!/bin/bash
#SBATCH --job-name=viral_inference
#SBATCH --output=logs/viral_inference_%j.out
#SBATCH --error=logs/viral_inference_%j.err
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

python -c "import torch, transformers, llava; print(torch.__version__, transformers.__version__, llava.__file__)"

# -------------------------------
# HUGGINGFACE CACHE CONFIG
# -------------------------------
export HF_HOME=/cluster/project/cvg/students/fbondi/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets

python -c "from huggingface_hub import hf_hub_download; import os; print('HF cache dir:', os.getenv('HF_HOME'))"

# -------------------------------
# PATH CONFIGURATION
# -------------------------------
# Usa direttamente i repo Hugging Face, non più i checkpoint locali
MODEL_PATH="/cluster/project/cvg/students/fbondi/sem-project/VIRAL/checkpoints/viral_checkpoints/llava-v1.5-7b-instruct-repa-dino-single-16"
IMAGE_PATH="$PWD/images/web.jpg"
PROMPT="Describe the image in detail."

# -------------------------------
# INFERENCE COMMAND
# -------------------------------
echo "[INFO] Launching inference..."
python inference.py \
  --model-path "$MODEL_PATH" \
  --model-base "liuhaotian/llava-v1.5-7b" \
  --image-path "$IMAGE_PATH" \
  --prompt "$PROMPT" \
  --temperature 0.1 \
  --top_p 0.7 \
  --max-new-tokens 100

echo "[INFO] Job finished at $(date)"