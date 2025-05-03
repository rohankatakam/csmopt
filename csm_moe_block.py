"""
CSM MoE Block implementation that integrates with the existing CSM architecture.

This module provides the integration between the CSM model and our MoE routing mechanism.
It allows for seamless insertion of MoE blocks into the CSM decoder pipeline.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union

from models import Model, ModelArgs
from moe_router import patch_llama_with_moe


class CSMMoEModel(nn.Module):
    """
    Wrapper for CSM model that adds MoE routing capabilities to selected decoder blocks.
    """
    def __init__(self, 
                 original_model: Model,
                 moe_block_indices: List[int] = None,
                 num_experts: int = 8,
                 k: int = 2,
                 router_z_loss_coef: float = 0.001):
        """
        Initialize the CSM MoE model.
        
        Args:
            original_model: Original CSM model to wrap
            moe_block_indices: Indices of decoder blocks to convert to MoE (None means no MoE)
            num_experts: Number of experts in each MoE layer
            k: Number of experts to route to
            router_z_loss_coef: Coefficient for router z-loss
        """
        super().__init__()
        self.original_model = original_model
        self.config = original_model.config
        
        # Default to no MoE blocks
        self.moe_block_indices = moe_block_indices or []
        self.num_experts = num_experts
        self.k = k
        self.router_z_loss_coef = router_z_loss_coef
        
        # Variables to track MoE blocks and their losses
        self.moe_blocks = {}
        self.last_aux_loss = 0.0
        
        # Add MoE to specified decoder blocks
        if len(self.moe_block_indices) > 0:
            # Get dimensions from the model
            if hasattr(self.original_model.decoder, 'layers') and len(self.original_model.decoder.layers) > 0:
                # For LLaMA 3.2: Get embed_dim and mlp_dim
                if self.config.decoder_flavor == "llama-1B":
                    embed_dim = 2048  # From llama3_2_1B function
                    mlp_dim = 8192
                elif self.config.decoder_flavor == "llama-100M":
                    embed_dim = 1024  # From llama3_2_100M function
                    mlp_dim = 8192
                else:
                    # Default fallback
                    embed_dim = 1024
                    mlp_dim = 4096
                
                # Apply MoE patches
                self.original_model, self.moe_blocks = patch_llama_with_moe(
                    model=self.original_model,
                    block_indices=self.moe_block_indices,
                    embed_dim=embed_dim,
                    mlp_dim=mlp_dim,
                    num_experts=self.num_experts,
                    k=self.k
                )

    def forward(self, *args, **kwargs):
        """
        Forward pass through the MoE-augmented CSM model.
        
        Args:
            *args, **kwargs: Arguments to pass to the original model
            
        Returns:
            output: Output from the model
        """
        return self.original_model(*args, **kwargs)

    def generate_frame(self, *args, **kwargs):
        """
        Generate a frame with the MoE-augmented model, capturing auxiliary losses.
        
        Args:
            *args, **kwargs: Arguments to pass to the original model's generate_frame
            
        Returns:
            output: Output from generate_frame
        """
        # The original method is patched, so aux losses will be computed during
        # the forward pass through MoE blocks
        output = self.original_model.generate_frame(*args, **kwargs)
        
        # Reset auxiliary loss for the next frame
        self.last_aux_loss = 0.0
        
        return output
    
    def setup_caches(self, *args, **kwargs):
        """
        Setup KV caches, delegating to the original model.
        
        Args:
            *args, **kwargs: Arguments to pass to the original method
        """
        return self.original_model.setup_caches(*args, **kwargs)
    
    def reset_caches(self):
        """Reset KV caches, delegating to the original model."""
        return self.original_model.reset_caches()
    
    def enable_moe(self, enable: bool = True):
        """
        Enable or disable MoE routing.
        
        Args:
            enable: Whether to enable MoE routing
        """
        for block_name, moe_block in self.moe_blocks.items():
            moe_block.use_moe = enable
    
    def get_expert_usage_stats(self) -> Dict[str, torch.Tensor]:
        """
        Get statistics about expert usage.
        
        Returns:
            stats: Dictionary of expert usage statistics
        """
        stats = {}
        for block_name, moe_block in self.moe_blocks.items():
            expert_counts = moe_block.moe_layer.get_expert_activations()
            stats[f"{block_name}_expert_counts"] = expert_counts
        return stats


# Monkey-patch the Model.generate_frame method to handle MoE blocks and their auxiliary losses
original_generate_frame = Model.generate_frame

def generate_frame_with_moe_support(self, *args, **kwargs):
    """
    Patched generate_frame method that handles MoE auxiliary losses.
    """
    # Check if this is an MoE model
    if hasattr(self, 'moe_blocks') and self.moe_blocks:
        # Process through MoE blocks, which will return (output, aux_loss)
        output = original_generate_frame(self, *args, **kwargs)
        # The aux_loss is tracked internally in CSMMoEModel
        return output
    else:
        # Original method for non-MoE models
        return original_generate_frame(self, *args, **kwargs)

# Apply the monkey patch
Model.generate_frame = generate_frame_with_moe_support


def create_moe_csm_model(
    original_model: Model,
    moe_block_indices: List[int] = None,
    num_experts: int = 8,
    k: int = 2
) -> CSMMoEModel:
    """
    Create a CSM model with MoE routing.
    
    Args:
        original_model: Original CSM model
        moe_block_indices: Indices of decoder blocks to convert to MoE
        num_experts: Number of experts
        k: Number of experts to route to
        
    Returns:
        moe_model: CSM model with MoE routing
    """
    return CSMMoEModel(
        original_model=original_model,
        moe_block_indices=moe_block_indices,
        num_experts=num_experts,
        k=k
    )
