#!/bin/bash
#SBATCH --job-name=dioporco
#SBATCH --output=logs/dioporco_%j.out
#SBATCH --error=logs/dioporco_%j.err
#SBATCH --time=00:05:00
#SBATCH --gpus=nvidia_geforce_rtx_4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G

module purge
module load eth_proxy || true
module load stack/2024-06
module load gcc/12.2.0
module load cuda/12.1.1

source /cluster/project/cvg/students/fbondi/miniconda3/etc/profile.d/conda.sh
conda activate viral

echo "=== Checking package versions ==="
echo ""

# Array di pacchetti con versioni richieste
declare -A required_versions=(
    ["torch"]="2.1.2"
    ["torchvision"]="0.16.2"
    ["transformers"]="4.37.2"
    ["tokenizers"]="0.15.1"
    ["sentencepiece"]="0.1.99"
    ["accelerate"]="0.21.0"
    ["scikit-learn"]="1.2.2"
    ["gradio"]="4.16.0"
    ["gradio_client"]="0.8.1"
    ["httpx"]="0.24.0"
    ["einops"]="0.6.1"
    ["einops-exts"]="0.0.4"
    ["timm"]="0.6.13"
)

all_match=true

for package in "${!required_versions[@]}"; do
    required="${required_versions[$package]}"
    installed=$(pip show "$package" 2>/dev/null | grep "^Version:" | awk '{print $2}')
    
    if [ -z "$installed" ]; then
        echo "❌ $package: NOT INSTALLED (required: $required)"
        all_match=false
    elif [ "$installed" == "$required" ]; then
        echo "✅ $package: $installed"
    else
        echo "⚠️  $package: $installed (required: $required)"
        all_match=false
    fi
done

echo ""
if [ "$all_match" = true ]; then
    echo "✅ All versions match!"
else
    echo "⚠️  Some versions don't match. Run:"
    echo "    pip install torch==2.1.2 torchvision==0.16.2 transformers==4.37.2 tokenizers==0.15.1 sentencepiece==0.1.99 accelerate==0.21.0 scikit-learn==1.2.2 gradio==4.16.0 gradio_client==0.8.1 httpx==0.24.0 einops==0.6.1 einops-exts==0.0.4 timm==0.6.13"
fi