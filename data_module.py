"""
PyTorch Lightning DataModule for Llama 4 to CSM adapter training.
Optimized for distributed training on multi-GPU systems.
"""
import os
import torch
from typing import Optional, Dict, List, Tuple
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split

class AdapterDataset(Dataset):
    """Dataset for training the Llama 4 adapter"""
    def __init__(self, pairs_path: str):
        self.data = torch.load(pairs_path)
        self.llama4_states = self.data["llama4_states"]
        self.csm_inputs = self.data["csm_inputs"]
        assert len(self.llama4_states) == len(self.csm_inputs)
        print(f"Loaded dataset with {len(self)} samples")
        
    def __len__(self) -> int:
        return len(self.llama4_states)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.llama4_states[idx], self.csm_inputs[idx]


class AdapterDataModule(pl.LightningDataModule):
    """Data module for Llama 4 adapter training with proper distributed handling"""
    def __init__(
        self,
        pairs_file: str,
        batch_size: int = 32,
        val_split: float = 0.1,
        num_workers: int = 8,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.pairs_file = pairs_file
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
    def setup(self, stage: Optional[str] = None):
        # Load the full dataset
        full_dataset = AdapterDataset(self.pairs_file)
        
        # Split into train and validation sets
        val_size = int(self.val_split * len(full_dataset))
        train_size = len(full_dataset) - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        print(f"Dataset split: {train_size} training samples, {val_size} validation samples")
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            drop_last=True,  # Drop last incomplete batch for more efficient training
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )
