#!/usr/bin/env python3
"""
Distributed training launcher for the Llama 4 adapter on multi-GPU systems.
Optimized for 8x H100 GPUs.
"""
import os
import sys
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the adapter trainer - adjust if your module structure is different
from train_adapter import AdapterTrainer, AdapterDataModule

def main():
    parser = argparse.ArgumentParser(description="Train Llama 4 adapter in distributed mode")
    
    # Data arguments
    parser.add_argument("--pairs_file", type=str, required=True, 
                        help="Path to the dataset file (.pt) containing input-output pairs")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio (default: 0.1)")
    
    # Model arguments
    parser.add_argument("--input_dim", type=int, default=8192,
                        help="Input dimension from Llama 4 (default: 8192)")
    parser.add_argument("--output_dim", type=int, default=4096,
                        help="Output dimension for CSM (default: 4096)")
    parser.add_argument("--hidden_dim", type=int, default=4096,
                        help="Hidden dimension in adapter (default: 4096)")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Training batch size per GPU (default: 512)")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate (default: 5e-5)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs (default: 5)")
    parser.add_argument("--precision", type=str, default="bf16", choices=["32", "16", "bf16"],
                        help="Training precision (default: bf16)")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps (default: 1)")
    
    # Output arguments
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints (default: checkpoints)")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory to save logs (default: logs)")
    
    # Performance arguments
    parser.add_argument("--num_nodes", type=int, default=1,
                        help="Number of nodes for distributed training (default: 1)")
    parser.add_argument("--num_gpus", type=int, default=8,
                        help="Number of GPUs per node (default: 8)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of data loader workers (default: 8)")
    
    args = parser.parse_args()
    
    # Configure path for checkpoints
    checkpoint_dir = os.path.join(args.checkpoint_dir, "adapter")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Configure checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="adapter-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    
    # Configure learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    # Configure logger
    logger = TensorBoardLogger(args.log_dir, name="adapter_training")
    
    # Configure trainer with DDP strategy
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=args.num_gpus,
        num_nodes=args.num_nodes,
        strategy=DDPStrategy(find_unused_parameters=False, gradient_as_bucket_view=True),
        precision=args.precision,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        accumulate_grad_batches=args.grad_accum,
        gradient_clip_val=1.0,
    )
    
    # Create data module
    data_module = AdapterDataModule(
        pairs_file=args.pairs_file,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.workers,
    )
    
    # Create model
    model = AdapterTrainer(
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
    )
    
    # Train the model
    trainer.fit(model, data_module)
    
    print(f"Training complete! Best model saved at: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()
