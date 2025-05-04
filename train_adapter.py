import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser
from typing import Tuple, Dict, List, Optional

from llama4_adapter import Llama4Adapter

class Llama4AdapterDataset(Dataset):
    """Dataset for training the Llama 4 adapter"""
    def __init__(self, pairs_path: str):
        self.data = torch.load(pairs_path)
        self.llama4_states = self.data["llama4_states"]
        self.csm_inputs = self.data["csm_inputs"]
        assert len(self.llama4_states) == len(self.csm_inputs)
        
    def __len__(self) -> int:
        return len(self.llama4_states)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.llama4_states[idx], self.csm_inputs[idx]

class Llama4AdapterTrainer(pl.LightningModule):
    """PyTorch Lightning module for training the adapter"""
    def __init__(self, 
                 input_dim: int = 8192, 
                 output_dim: int = 4096,
                 hidden_dim: Optional[int] = None,
                 learning_rate: float = 5e-4,
                 weight_decay: float = 0.01,
                 cosine_weight: float = 0.5):
        super().__init__()
        self.save_hyperparameters()
        
        # Adapter model
        self.model = Llama4Adapter(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim
        )
        
        # Loss weights
        self.cosine_weight = cosine_weight
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def mse_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)
    
    def cosine_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        
        # Combined loss: MSE + cosine similarity
        mse_loss = self.mse_loss(y_hat, y)
        cos_loss = self.cosine_loss(y_hat, y)
        loss = (1.0 - self.cosine_weight) * mse_loss + self.cosine_weight * cos_loss
        
        self.log("train_loss", loss)
        self.log("train_mse", mse_loss)
        self.log("train_cos", cos_loss)
        
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        y_hat = self(x)
        
        mse_loss = self.mse_loss(y_hat, y)
        cos_loss = self.cosine_loss(y_hat, y)
        loss = (1.0 - self.cosine_weight) * mse_loss + self.cosine_weight * cos_loss
        
        self.log("val_loss", loss)
        self.log("val_mse", mse_loss)
        self.log("val_cos", cos_loss)
        
        # Calculate cosine similarity (higher is better)
        cos_sim = F.cosine_similarity(y_hat, y, dim=-1).mean()
        self.log("val_cos_sim", cos_sim)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

def main():
    parser = ArgumentParser()
    parser.add_argument("--pairs", type=str, required=True, help="Path to pairs.pt file")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    parser.add_argument("--cosine_weight", type=float, default=0.5, 
                        help="Weight for cosine similarity loss (0-1)")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    full_dataset = Llama4AdapterDataset(args.pairs)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = Llama4AdapterTrainer(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        cosine_weight=args.cosine_weight
    )
    
    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="llama4_adapter-{epoch:02d}-{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min"
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        precision="16-mixed",  # Use mixed precision for faster training
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Save final model
    final_path = os.path.join(args.output_dir, "adapter.ckpt")
    trainer.save_checkpoint(final_path)
    print(f"Final model saved to {final_path}")
    
    # Save lightweight version (just the adapter, not the trainer)
    adapter_path = os.path.join(args.output_dir, "adapter_only.pt")
    torch.save(model.model.state_dict(), adapter_path)
    print(f"Adapter-only model saved to {adapter_path}")

if __name__ == "__main__":
    main()
