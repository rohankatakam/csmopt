"""
Mixed precision support for MoE optimization.

This module provides utilities for mixed precision inference,
allowing the model to leverage faster computation with lower precision
while maintaining numerical stability.
"""

import torch
import torch.nn as nn
from typing import Union, Dict, Any, Optional, Tuple


class MixedPrecisionManager:
    """
    Manages precision conversion for MoE operations.
    
    This class handles automatic conversion between precision types 
    for faster computation with minimal accuracy loss.
    """
    
    def __init__(
        self, 
        precision: str = "fp16", 
        calibration_factor: float = 1.0,
        dynamic_scale: bool = True
    ):
        """
        Initialize the mixed precision manager.
        
        Args:
            precision: Precision type to use ("fp32", "fp16", "bf16")
            calibration_factor: Factor to scale values for numerical stability
            dynamic_scale: Whether to dynamically adjust scaling based on tensor values
        """
        self.precision = precision
        self.calibration_factor = calibration_factor
        self.dynamic_scale = dynamic_scale
        
        # Check if bfloat16 is available (requires recent PyTorch and NVIDIA Ampere+ GPUs)
        self.bf16_available = torch.cuda.is_available() and hasattr(torch, 'bfloat16')
        
        # Determine compute_dtype and storage_dtype based on precision setting
        if precision == "fp32":
            self.compute_dtype = torch.float32
            self.storage_dtype = torch.float32
        elif precision == "bf16" and self.bf16_available:
            self.compute_dtype = torch.bfloat16
            self.storage_dtype = torch.bfloat16
        else:  # Default to fp16 if precision isn't recognized or bf16 isn't available
            if precision == "bf16" and not self.bf16_available:
                print("Warning: bfloat16 not available, falling back to float16")
            self.compute_dtype = torch.float16
            self.storage_dtype = torch.float16
        
        self.current_scale = calibration_factor
        
    def get_compute_dtype(self) -> torch.dtype:
        """Get the compute data type."""
        return self.compute_dtype
    
    def get_storage_dtype(self) -> torch.dtype:
        """Get the storage data type."""
        return self.storage_dtype
    
    def to_compute_precision(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert tensor to compute precision.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Tensor in compute precision
        """
        if tensor.dtype == self.compute_dtype:
            return tensor
        
        return tensor.to(dtype=self.compute_dtype)
    
    def to_storage_precision(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert tensor to storage precision.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Tensor in storage precision
        """
        if tensor.dtype == self.storage_dtype:
            return tensor
            
        return tensor.to(dtype=self.storage_dtype)
    
    def apply_dynamic_scaling(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Apply dynamic scaling to prevent underflow/overflow.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Tuple of (scaled tensor, scale factor)
        """
        if not self.dynamic_scale or tensor.dtype == torch.float32:
            return tensor, 1.0
            
        # Check tensor range to determine if scaling is needed
        max_val = torch.max(torch.abs(tensor)).item()
        
        # Only scale if needed to prevent underflow/overflow
        if max_val > 0:
            if max_val > 65000:  # Close to fp16 max
                scale = 65000 / max_val
                return tensor * scale, scale
            elif max_val < 1e-4:  # Prevent underflow
                scale = 1e-4 / max_val
                return tensor * scale, scale
                
        return tensor, 1.0
    
    def manage_router_precision(
        self, 
        logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Manage precision for router logits, with special handling for numeric stability.
        
        Args:
            logits: Router logits
            
        Returns:
            Tuple of (routing_weights, expert_indices) in appropriate precision
        """
        # Convert to compute precision for operations
        logits = self.to_compute_precision(logits)
        
        # Apply appropriate normalization based on precision
        if self.compute_dtype == torch.float16:
            # Special handling for fp16 to avoid overflow
            logits_scaled, _ = self.apply_dynamic_scaling(logits)
            
            # Get top-k values and indices
            top_k_weights, top_k_indices = torch.topk(
                logits_scaled, k=2, dim=-1
            )
            
            # Apply softmax with better numerical stability
            max_weights = torch.max(top_k_weights, dim=-1, keepdim=True)[0]
            top_k_weights = top_k_weights - max_weights
            top_k_weights = torch.exp(top_k_weights)
            sum_weights = torch.sum(top_k_weights, dim=-1, keepdim=True)
            weights = top_k_weights / sum_weights
        else:
            # For fp32/bf16, we can directly use top-k + softmax
            top_k_weights, top_k_indices = torch.topk(
                logits, k=2, dim=-1
            )
            weights = torch.nn.functional.softmax(top_k_weights, dim=-1)
        
        # Return in storage precision
        return self.to_storage_precision(weights), top_k_indices
    
    def apply_to_module(self, module: nn.Module) -> nn.Module:
        """
        Apply mixed precision settings to a module.
        
        Args:
            module: PyTorch module to convert
            
        Returns:
            Module with updated precision
        """
        if self.precision == "fp32":
            return module  # No conversion needed
            
        # Convert parameters
        for param in module.parameters():
            param.data = param.data.to(self.storage_dtype)
            
        return module


def convert_moe_model_to_mixed_precision(
    model: nn.Module, 
    precision: str = "fp16"
) -> Tuple[nn.Module, MixedPrecisionManager]:
    """
    Convert a MoE model to use mixed precision.
    
    Args:
        model: PyTorch model with MoE layers
        precision: Precision to use ("fp32", "fp16", "bf16")
        
    Returns:
        Tuple of (converted model, precision manager)
    """
    # Create precision manager
    precision_manager = MixedPrecisionManager(precision=precision)
    
    # No conversion needed for fp32
    if precision == "fp32":
        return model, precision_manager
    
    print(f"Converting MoE modules to {precision}...")
    
    # For CSM MoE models, we need special handling
    if hasattr(model, "moe_blocks"):
        for block_name, moe_block in model.moe_blocks.items():
            print(f"  Converting {block_name} to {precision}")
            
            # Convert MoE layer
            if hasattr(moe_block, "moe"):
                # Convert router directly first
                if hasattr(moe_block.moe, "router"):
                    for param in moe_block.moe.router.parameters():
                        param.data = param.data.to(precision_manager.get_storage_dtype())
                
                # Convert experts
                if hasattr(moe_block.moe, "experts"):
                    for i, expert in enumerate(moe_block.moe.experts):
                        for param in expert.parameters():
                            param.data = param.data.to(precision_manager.get_storage_dtype())
    
    # Attach precision manager to the model
    if not hasattr(model, "precision_manager"):
        model.precision_manager = precision_manager
    
    return model, precision_manager
