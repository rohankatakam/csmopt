"""
CSM MoE Block implementation that integrates with the existing CSM architecture.

This module provides the integration between the CSM model and our optimized vectorized MoE 
routing mechanism that improves memory usage and performance.
It allows for seamless insertion of MoE blocks into the CSM decoder pipeline.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import time

from models import Model, ModelArgs
from optimized_moe import OptimizedMoEBlockWrapper, VectorizedMoELayer


class CSMMoEModel(nn.Module):
    """
    Wrapper for CSM model that adds vectorized MoE routing capabilities to selected decoder blocks.
    This implementation optimizes for memory usage and performance.
    """
    def __init__(self, 
                 original_model: Model,
                 moe_block_indices: List[int] = None,
                 num_experts: int = 8,
                 top_k: int = 2,
                 use_vectorized: bool = True):
        """
        Initialize the CSM MoE model with vectorized implementation.
        
        Args:
            original_model: Original CSM model to wrap
            moe_block_indices: Indices of decoder blocks to convert to MoE (None means no MoE)
            num_experts: Number of experts in each MoE layer
            top_k: Number of experts to route to
            use_vectorized: Whether to use the vectorized implementation
        """
        super().__init__()
        self.original_model = original_model
        self.config = original_model.config
        
        # Default to no MoE blocks
        self.moe_block_indices = moe_block_indices or []
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_vectorized = use_vectorized
        
        # Variables to track MoE blocks
        self.moe_blocks = {}
        
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
                
                # Apply vectorized MoE to specified decoder blocks
                self.moe_blocks = self.patch_llama_with_vectorized_moe(
                    embed_dim=embed_dim,
                    mlp_dim=mlp_dim
                )

    def patch_llama_with_vectorized_moe(self, embed_dim, mlp_dim):
        """
        Apply vectorized MoE to the specified decoder blocks.
        
        Args:
            embed_dim: Embedding dimension
            mlp_dim: MLP dimension
            
        Returns:
            moe_blocks: Dictionary of MoE blocks for monitoring
        """
        moe_blocks = {}
        
        # Get decoder layers
        decoder = self.original_model.decoder
        
        # Track replacement time for performance benchmarking
        start_time = time.time()
        
        # Wrap specified blocks with Vectorized MoE
        for idx in self.moe_block_indices:
            # Check if the index is valid
            if hasattr(decoder, 'layers') and idx < len(decoder.layers):
                original_block = decoder.layers[idx]
                
                # Create optimized MoE block wrapper
                moe_block = OptimizedMoEBlockWrapper(
                    original_block=original_block,
                    hidden_dim=embed_dim,
                    mlp_dim=mlp_dim,
                    num_experts=self.num_experts,
                    top_k=self.top_k
                )
                
                # Replace the original block
                decoder.layers[idx] = moe_block
                moe_blocks[f"decoder_block_{idx}"] = moe_block
        
        elapsed = time.time() - start_time
        print(f"Patched {len(self.moe_block_indices)} decoder blocks with vectorized MoE in {elapsed:.2f}s")
        
        return moe_blocks
    
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
        Generate a frame with the MoE-augmented model.
        
        Args:
            *args, **kwargs: Arguments to pass to the original model's generate_frame
            
        Returns:
            output: Output from generate_frame
        """
        # Vectorized implementation handles activation tracking internally
        output = self.original_model.generate_frame(*args, **kwargs)
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
    
    def get_expert_usage_stats(self) -> Dict[str, Dict]:
        """
        Get detailed statistics about expert usage for all MoE blocks.
        
        Returns:
            stats: Dictionary of expert usage statistics for each MoE block
        """
        stats = {}
        for block_name, moe_block in self.moe_blocks.items():
            stats[block_name] = moe_block.moe.get_expert_usage_stats()
        return stats


# Monkey-patch the Model.generate_frame method to handle vectorized MoE blocks
original_generate_frame = Model.generate_frame

def generate_frame_with_vectorized_moe_support(self, *args, **kwargs):
    """
    Patched generate_frame method that supports vectorized MoE operations.
    """
    # Check if this is an MoE model
    if hasattr(self, 'moe_blocks') and self.moe_blocks:
        # Process through vectorized MoE blocks
        output = original_generate_frame(self, *args, **kwargs)
        return output
    else:
        # Original method for non-MoE models
        return original_generate_frame(self, *args, **kwargs)

# Apply the monkey patch
Model.generate_frame = generate_frame_with_vectorized_moe_support


def create_moe_csm_model(
    original_model: Model,
    moe_block_indices: List[int] = None,
    num_experts: int = 8,
    top_k: int = 2,
    use_vectorized: bool = True
) -> CSMMoEModel:
    """
    Create a CSM model with vectorized MoE routing.
    
    Args:
        original_model: Original CSM model
        moe_block_indices: Indices of decoder blocks to convert to MoE
        num_experts: Number of experts
        top_k: Number of experts to route to
        use_vectorized: Whether to use the vectorized implementation
        
    Returns:
        moe_model: CSM model with vectorized MoE routing
    """
    return CSMMoEModel(
        original_model=original_model,
        moe_block_indices=moe_block_indices,
        num_experts=num_experts,
        top_k=top_k,
        use_vectorized=use_vectorized
    )
