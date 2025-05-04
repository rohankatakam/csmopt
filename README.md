# Llama 4 to CSM Text-to-Speech Integration

This repository implements a dimensional adapter for integrating Llama large language models with Sesame's Conversational Speech Model (CSM) for high-quality text-to-speech synthesis.

## Overview

The Llama 4 to CSM adapter provides a lightweight translation layer that converts embeddings from Llama models (5120d for Llama 4 Scout or 2048d for Llama-3.2-1B) to the 4096d format expected by the CSM decoder.

## Features

- **Dimensional Adapter**: Bridges between Llama hidden states and CSM input format
- **Model Flexibility**: Support for both Llama 4 Scout (5120d) and Llama-3.2-1B (2048d)
- **MoE Optimization**: Special optimizations for Mixture of Experts models 
- **Easy Inference**: Simple command-line interface for generating speech from text

## Requirements

- Python 3.10+
- CUDA-compatible GPU (16GB+ VRAM recommended)
- Access to the following Hugging Face models:
  - [Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) (primary model)
  - [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) (alternative model)
  - [CSM-1B](https://huggingface.co/sesame/csm-1b)
- Hugging Face API token with access to gated models

## Setup

```bash
# Clone repository
git clone https://github.com/your-organization/csmopt.git
cd csmopt

# Create conda environment
conda create -n csm_env python=3.10 -y
conda activate csm_env

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Set environment variables
export HUGGING_FACE_HUB_TOKEN=your_token_here
```

## Directory Structure

```
csmopt/
├── checkpoints/                # Trained adapter models
├── data/                       # Training data
├── docs/                       # Documentation
├── scripts/                    # Utility scripts
│   ├── generate_synthetic_data.py    # Generate training data
│   └── train_adapter.py              # Train adapter model
├── tests/                      # Testing scripts
├── generate_speech.py          # Fast inference script
├── llama4_adapter.py           # Adapter model definition
├── llama4_integration.py       # Llama integration code
└── train_adapter.py            # Simple adapter training
```

## Quick Start

### Generate Speech with Pre-trained Adapter

```bash
# Generate speech with Llama 4 Scout model
python generate_speech.py \
  --text "Your text to convert to speech." \
  --output output.wav \
  --adapter checkpoints/llama4_improved_adapter_adapter.pt

# Or use Llama-3.2-1B model (less resource intensive)
python generate_speech.py \
  --text "Your text to convert to speech." \
  --output output.wav \
  --model meta-llama/Llama-3.2-1B \
  --adapter checkpoints/adapter_only.pt
```

## Training Process

### 1. Generate Training Data

```bash
# For Llama 4 Scout (5120d)
python scripts/generate_synthetic_data.py \
  --output data/llama4_train_data.pt \
  --n_samples 20000 \
  --llama4_dim 5120 \
  --csm_dim 4096 \
  --noise_level 0.03

# For Llama-3.2-1B (2048d)
python scripts/generate_synthetic_data.py \
  --output data/llama3_train_data.pt \
  --n_samples 20000 \
  --llama4_dim 2048 \
  --csm_dim 4096 \
  --noise_level 0.03
```

### 2. Train the Adapter

```bash
# For Llama 4 Scout (5120d)
python scripts/train_adapter.py \
  --data_path data/llama4_train_data.pt \
  --input_dim 5120 \
  --output_dim 4096 \
  --epochs 15 \
  --batch_size 64 \
  --learning_rate 5e-4 \
  --cosine_weight 0.7 \
  --output_dir checkpoints \
  --experiment_name llama4_adapter

# For Llama-3.2-1B (2048d)
python scripts/train_adapter.py \
  --data_path data/llama3_train_data.pt \
  --input_dim 2048 \
  --output_dim 4096 \
  --epochs 15 \
  --batch_size 64 \
  --learning_rate 5e-4 \
  --cosine_weight 0.7 \
  --output_dir checkpoints \
  --experiment_name llama3_adapter
```

### 3. Test the Adapter

```bash
# Run integration test
python tests/test_integration_pipeline.py \
  --adapter checkpoints/llama4_adapter_adapter.pt \
  --model meta-llama/Llama-4-Scout-17B-16E-Instruct

# Generate speech with trained adapter
python generate_speech.py \
  --text "This is a test of the trained adapter." \
  --output test_output.wav \
  --adapter checkpoints/llama4_adapter_adapter.pt
```

## Advanced Options

### Using Real Llama 4 Embeddings

For higher quality training data, you can extract real embeddings from the Llama 4 model:

```bash
python scripts/generate_synthetic_data.py \
  --output data/llama4_real_data.pt \
  --n_samples 10000 \
  --llama4_dim 5120 \
  --csm_dim 4096 \
  --noise_level 0.05 \
  --use_real_llama4 \
  --llama4_model meta-llama/Llama-4-Scout-17B-16E-Instruct
```

Note: This requires significant GPU memory and may take a long time to complete.

### Adapter Hyperparameters

The adapter quality can be tuned with these hyperparameters:

- `--cosine_weight`: Weight for cosine similarity loss (higher preserves directional relationships)
- `--hidden_dim`: Hidden layer dimension (larger can increase adapter capacity)
- `--l1_weight`: Weight for L1 regularization (can help with adapter generalization)

## Troubleshooting

### Out of Memory Errors

If you encounter CUDA out of memory errors:
- Use Llama-3.2-1B instead of Llama 4 Scout (much smaller model)
- Reduce batch size for training
- Enable gradient checkpointing
- Use 4-bit quantization for Llama model loading

### Audio Quality Issues

If you experience audio quality degradation:
- Try lowering the `noise_level` parameter when generating training data
- Increase the `cosine_weight` during training
- Train for more epochs
- Generate more training samples

## License

[License information here]

## Acknowledgments

- Sesame AI for the CSM model
- Meta AI for the Llama models
