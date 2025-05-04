#!/bin/bash
# Main setup script for CSM + Llama 4 integration
set -e  # Exit on any error

echo "===== Llama 4 to CSM Integration Setup ====="
echo "This script will set up the environment and required dependencies."

# Check if token is provided as argument
if [ -n "$1" ]; then
    export HUGGING_FACE_HUB_TOKEN="$1"
    echo "Using provided Hugging Face token."
else
    # Prompt for token if not provided
    echo ""
    echo "You need a Hugging Face token with access to:"
    echo "- meta-llama/Llama-4-Scout-17B-16E-Instruct"
    echo "- meta-llama/Llama-3.2-1B"
    echo "- sesame/csm-1b"
    echo ""
    read -p "Enter your Hugging Face token (or press Enter to skip): " token
    
    if [ -n "$token" ]; then
        export HUGGING_FACE_HUB_TOKEN="$token"
        echo "Token set successfully."
    else
        echo "No token provided. You'll need to set it manually later with:"
        echo "export HUGGING_FACE_HUB_TOKEN=your_token_here"
    fi
fi

# Pass the token to the environment setup script
echo "Running environment setup..."
export HF_TOKEN="$HUGGING_FACE_HUB_TOKEN"
bash scripts/setup_environment.sh

# Login to Hugging Face
if [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "Logging in to Hugging Face..."
    python -c "import os; from huggingface_hub import login; login(token=os.environ['HUGGING_FACE_HUB_TOKEN'])"
fi

echo ""
echo "===== Setup Complete ====="
echo "To use this environment:"
echo "1. Activate it with: source ~/miniconda3/bin/activate csm_fixed"
echo "2. Set your token: export HUGGING_FACE_HUB_TOKEN=your_token_here"
echo "3. Run tests: python scripts/test_integration_pipeline.py"
echo ""
echo "For a full test with real models, run:"
echo "python scripts/run_real_tts_test.py --token \$HUGGING_FACE_HUB_TOKEN --model meta-llama/Llama-3.2-1B"
