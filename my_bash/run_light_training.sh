#!/bin/bash
#SBATCH --job-name=train_with_cl
#SBATCH --output=logs/train_with_cl_%j.out
#SBATCH --error=logs/train_with_cl_%j.err
#SBATCH --time=10:00:00
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=gpu

# Attiva Conda
source /cluster/project/cvg/students/fbondi/miniconda3/etc/profile.d/conda.sh
conda activate viral

module purge
module load eth_proxy || true
module load stack/2024-06
module load gcc/12.2.0
module load cuda/12.1.1

python - <<'EOF'
import sys, torch, subprocess, os
print("🧩 Python version:", sys.version.split()[0])
print("🔥 PyTorch version:", torch.__version__)
print("🎮 CUDA (PyTorch):", torch.version.cuda)
print("🧠 Torch built with:", torch.__config__.show())

# Check CUDA toolkit (nvcc)
nvcc = subprocess.run(["which", "nvcc"], capture_output=True, text=True)
print("🔧 nvcc path:", nvcc.stdout.strip() or "Not found")
subprocess.run(["nvcc", "--version"])

# Check ABI (important for FlashAttention)
try:
    abi = subprocess.run(["g++", "-dM", "-E", "-x", "c++", "-", "-v"], input="", capture_output=True, text=True)
    for line in abi.stdout.splitlines():
        if "GLIBCXX_USE_CXX11_ABI" in line:
            print("⚙️ CXX11 ABI flag:", line.strip())
            break
except Exception as e:
    print("⚠️ Could not check ABI:", e)
EOF

echo "-------------------------------"

# Optional: safety check for FlashAttention
if ! python -c "import flash_attn" &> /dev/null; then
    echo "[SETUP] Installing FlashAttention2..."
    pip uninstall -y flash-attn
    pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
fi

nvidia-smi
python -c "import flash_attn; print('FlashAttention version:', flash_attn.__version__)" || echo "[WARNING] FlashAttention not installed properly."
python -c "import torch, transformers, llava; print(torch.__version__, transformers.__version__, llava.__file__)"

# -------------------------------
# HUGGINGFACE CACHE CONFIG
# -------------------------------
export HF_HOME=/cluster/project/cvg/students/fbondi/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets

export FLASH_ATTENTION_DISABLE=1
export WANDB_API_KEY='146b803050dcb2863bd4cfa57a18b359bc33717e'
export WANDB_PROJECT="SEMESTER-PROJECT"
export WANDB_ENTITY="francesco-bondi02-eth-z-rich"
export WANDB_SILENT=true
export PYTHONPATH=/cluster/project/cvg/students/fbondi/sem-project/VIRAL:$PYTHONPATH

python -c "from huggingface_hub import hf_hub_download; import os; print('HF cache dir:', os.getenv('HF_HOME'))"

# ------------------------------------------------------------
# Light sanity-check training for VIRAL (single A100 GPU)
# ------------------------------------------------------------

echo "[INFO] Starting lightweight VIRAL training test on A100..."

cd /cluster/project/cvg/students/fbondi/sem-project/VIRAL
deepspeed --include localhost:0 --master_port=29503 llava/train/train_mem.py \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 128 \
    --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./llava_v1_5_mix665k.json \
    --image_folder /cluster/project/cvg/students/fbondi/datasets \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-custom-trained \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --save_strategy "no" \
    --evaluation_strategy "no" \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --config_path ./config.json

echo "[INFO] Lightweight training test finished on A100!"