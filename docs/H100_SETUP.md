# Llama 4 + CSM Integration: 8x H100 Setup Guide

This guide provides instructions for setting up and running the Llama 4 to CSM adapter training on an 8x H100 instance.

## 1. Environment Setup

After connecting to your instance, set up the environment with our automated script:

```bash
# Make the script executable
chmod +x scripts/setup_environment.sh

# Run the setup script
./scripts/setup_environment.sh

# Activate the environment
source ~/miniconda3/bin/activate csm_llama4
```

## 2. Code Transfer

Transfer the codebase to your instance using one of these methods:

**Option 1: Git (recommended if using a repository)**
```bash
git clone your-repository-url
cd your-repository-directory
```

**Option 2: SCP/RSYNC (for direct transfer)**
```bash
# From your local machine:
rsync -avz --exclude 'env/' --exclude '*.pt' --exclude '*.ckpt' /path/to/local/csm/ username@your-instance-ip:~/csm/
```

## 3. Model Access Setup

Ensure you have Hugging Face access to the required models:

```bash
# Log in to Hugging Face
huggingface-cli login
# Enter your token when prompted
```

## 4. Generating Training Data

If you haven't already generated the training data pairs:

```bash
# Generate 50k pairs (adjust as needed)
python scripts/dump_pairs.py --n 50000 --output data/pairs.pt
```

## 5. Distributed Training

Run training using our optimized distributed script:

```bash
# For single node, 8x GPUs (most common setup)
python scripts/train_distributed.py \
  --pairs_file data/pairs.pt \
  --batch_size 512 \
  --epochs 5 \
  --precision bf16 \
  --checkpoint_dir checkpoints \
  --log_dir logs

# For even faster training with gradient accumulation
python scripts/train_distributed.py \
  --pairs_file data/pairs.pt \
  --batch_size 256 \
  --grad_accum 2 \
  --epochs 5 \
  --precision bf16
```

## 6. Monitoring Training

Track training progress with TensorBoard:

```bash
tensorboard --logdir logs --bind_all
# Access via your browser at http://your-instance-ip:6006
```

## 7. Testing the Trained Adapter

Once training is complete, test the adapter:

```bash
python run_csm_moe.py --model llama4 --adapter checkpoints/adapter/adapter-epoch=XX-val_loss=X.XXXX.ckpt
```

## 8. Performance Optimization Tips

For maximum performance on the 8x H100 system:

1. **Precision**: Always use `bf16` precision for H100s (faster than fp16)
2. **Batch Size**: Start with 256-512 per GPU and adjust as needed
3. **System Settings**: Set `CUDA_VISIBLE_DEVICES` if you want to use specific GPUs
4. **Memory Management**: Use `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` for better memory management

## 9. Troubleshooting

Common issues and solutions:

* **OOM Errors**: Reduce batch size or enable gradient accumulation
* **Slow Data Loading**: Increase number of workers (try 8-16)
* **GPU Underutilization**: Enable profiling with `--profile` to diagnose bottlenecks

## 10. File Structure Reference

```
csm/
├── llama4_adapter.py      # Adapter model architecture
├── train_adapter.py       # Base training script
├── data_module.py         # Data loading for distributed training
├── generator.py           # Generator with Llama 4 support
├── run_csm_moe.py         # Main inference script
├── scripts/
│   ├── setup_environment.sh   # Environment setup script
│   ├── train_distributed.py   # Distributed training launcher
│   ├── dump_pairs.py          # Data generation script
│   └── quick_test_llama4.py   # Llama 4 access test script
└── checkpoints/           # Saved model checkpoints
```
