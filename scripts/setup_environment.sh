#!/bin/bash
# Setup script for Llama 4 + CSM environment on H100 instances
set -e  # Exit on any error

echo "Setting up environment for Llama 4 + CSM adapter training..."

# Create and activate conda environment
if [ ! -d "$HOME/miniconda3" ]; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
fi

# Add conda to path for this script
export PATH="$HOME/miniconda3/bin:$PATH"

# Create environment if it doesn't exist
if ! conda info --envs | grep -q "csm_llama4"; then
    echo "Creating conda environment..."
    conda create -n csm_llama4 python=3.10 -y
fi

# Activate environment
source $HOME/miniconda3/bin/activate csm_llama4

# Install all dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r "$(dirname "$0")/../requirements.txt"

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA..."
python -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Set environment variables for optimal performance
echo "Setting environment variables..."
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Begin downloading models in background
echo "Starting model downloads in background..."
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('meta-llama/Llama-4-Scout-17B-16E-Instruct'); print('Tokenizer downloaded')" &

echo "Environment setup complete!"
echo "To download the full Llama 4 model (this will take time):"
echo "python -c \"from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('meta-llama/Llama-4-Scout-17B-16E-Instruct', load_in_4bit=True, device_map='auto')\""
echo ""
echo "To use this environment: source $HOME/miniconda3/bin/activate csm_llama4"
