#!/usr/bin/env python3
"""
Optimized MoE implementation with vectorized operations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from typing import List, Tuple, Optional, Dict, Any
from moe_precision import MixedPrecisionManager
from optimized_router import OptimizedMoERouter

class VectorizedMoEGating(nn.Module):
    """
    Vectorized implementation of top-k gating for Mixture of Experts.
    """
    def __init__(self, input_dim: int, num_experts: int, top_k: int = 2, precision_manager: Optional[MixedPrecisionManager] = None):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(input_dim, num_experts, bias=False)
        
        # Initialize router weights 
        nn.init.normal_(self.router.weight, mean=0.0, std=0.1)
        
        # Mixed precision support
        self.precision_manager = precision_manager
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating network.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Tuple of (routing_weights, expert_indices)
            - routing_weights: Tensor of shape [batch_size, seq_len, top_k]
            - expert_indices: Tensor of shape [batch_size, seq_len, top_k]
        """
        # Get device from input
        device = x.device
        dtype = x.dtype
        
        # Ensure router is on the same device as input
        if self.router.weight.device != device:
            self.router.to(device=device, dtype=dtype)
        
        # Compute router logits - [batch_size, seq_len, num_experts]
        batch_size, seq_len, _ = x.shape
        
        # Flatten input for efficient routing
        x_flat = x.reshape(-1, self.input_dim)
        
        # Handle mixed precision for input if manager exists
        if self.precision_manager is not None:
            x_flat = self.precision_manager.to_compute_precision(x_flat)
            
        # Get router logits
        router_logits = self.router(x_flat)  # [batch_size*seq_len, num_experts]
        
        # Use precision manager for router operations if available
        if self.precision_manager is not None:
            routing_weights, expert_indices = self.precision_manager.manage_router_precision(router_logits)
            # Reshape back to original dimensions
            routing_weights = routing_weights.reshape(batch_size, seq_len, self.top_k)
            expert_indices = expert_indices.reshape(batch_size, seq_len, self.top_k)
        else:
            # Default implementation without precision management
            # Get top-k experts and weights
            routing_weights, expert_indices = torch.topk(
                router_logits, k=self.top_k, dim=-1
            )  # Both: [batch_size*seq_len, top_k]
            
            # Apply softmax to weights
            routing_weights = F.softmax(routing_weights, dim=-1)
            
            # Reshape back to original dimensions
            routing_weights = routing_weights.reshape(batch_size, seq_len, self.top_k)
            expert_indices = expert_indices.reshape(batch_size, seq_len, self.top_k)
        
        return routing_weights, expert_indices


class VectorizedMoEExpertLayer(nn.Module):
    """
    Expert layer for Mixture of Experts with optimized batch processing.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, precision_manager: Optional[MixedPrecisionManager] = None):
        super().__init__()
        self.up_proj = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.down_proj = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights for better performance
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
        nn.init.zeros_(self.down_proj.bias)
        
        # Mixed precision support
        self.precision_manager = precision_manager
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the expert.
        
        Args:
            x: Input tensor of any shape as long as last dim is input_dim
            
        Returns:
            Output tensor with same shape as input except last dim is output_dim
        """
        # Ensure expert is on the same device as input
        device = x.device
        dtype = x.dtype
        
        if self.up_proj.weight.device != device:
            self.to(device=device, dtype=dtype)
        
        # Handle mixed precision for computation
        if self.precision_manager is not None:
            # Convert to compute precision
            x = self.precision_manager.to_compute_precision(x)
            
            # Expert computation
            hidden = self.up_proj(x)
            hidden = self.act(hidden)
            out = self.down_proj(hidden)
            
            # Convert back to storage precision
            return self.precision_manager.to_storage_precision(out)
        else:
            # Standard computation path
            return self.down_proj(self.act(self.up_proj(x)))


class VectorizedMoELayer(nn.Module):
    """
    Vectorized MoE layer with improved batch processing.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_experts: int = 8, top_k: int = 2, precision: str = "fp32",
                 routing_algorithm: str = "top_k"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.routing_algorithm = routing_algorithm
        
        # Set up mixed precision support
        self.precision = precision
        self.precision_manager = None
        if precision != "fp32":
            self.precision_manager = MixedPrecisionManager(precision=precision)
        
        # Use the optimized router instead of the simple gating mechanism
        self.router = OptimizedMoERouter(
            input_dim=input_dim,
            num_experts=num_experts,
            top_k=top_k,
            routing_algorithm=routing_algorithm
        )
        
        # Create experts with precision manager
        self.create_experts()
        
        # Track expert usage for load balancing
        self.register_buffer(
            "expert_activation_count", 
            torch.zeros(num_experts, dtype=torch.int32),
            persistent=False
        )
        
        # Track token processing
        self.total_tokens_processed = 0
        
        # Load balancing auxiliary loss coefficient
        self.load_balancing_coeff = 0.01
    
    def create_experts(self):
        """Create the expert modules based on current configuration."""
        self.experts = nn.ModuleList([
            VectorizedMoEExpertLayer(self.input_dim, self.hidden_dim, self.output_dim, self.precision_manager)
            for _ in range(self.num_experts)
        ])
        
        # Reset the activation counter to match the new expert count
        self.register_buffer(
            "expert_activation_count", 
            torch.zeros(self.num_experts, dtype=torch.int32),
            persistent=False
        )
    
    def set_routing_algorithm(self, algorithm: str):
        """Set the routing algorithm."""
        if hasattr(self.router, 'set_routing_algorithm'):
            self.router.set_routing_algorithm(algorithm)
            self.routing_algorithm = algorithm
    
    def change_num_experts(self, num_experts: int):
        """Change the number of experts dynamically."""
        if num_experts == self.num_experts:
            return
            
        print(f"Changing number of experts from {self.num_experts} to {num_experts}")
        old_experts = self.num_experts
        self.num_experts = num_experts
        
        # Create new router with updated number of experts
        self.router = OptimizedMoERouter(
            input_dim=self.input_dim,
            num_experts=num_experts,
            top_k=min(self.top_k, num_experts),  # Ensure top_k isn't larger than num_experts
            routing_algorithm=self.routing_algorithm
        )
        
        # Update experts
        self.create_experts()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Forward pass through the MoE layer with vectorized operations.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Tuple of (output, aux_loss)
            - output: Tensor of shape [batch_size, seq_len, output_dim]
            - aux_loss: Auxiliary load balancing loss
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        input_dtype = x.dtype
        
        # Save original input dtype to ensure consistent output type
        # This is important when the MoE layer is used within a model that has its own dtype expectations
        output_dtype = input_dtype
        
        # For mixed precision, we handle the input conversion carefully
        compute_dtype = input_dtype
        if self.precision_manager is not None:
            compute_dtype = self.precision_manager.get_compute_dtype()
            # Only convert if necessary
            if x.dtype != compute_dtype:
                x = x.to(dtype=compute_dtype)
        
        # Get routing weights, expert indices, and routing aux loss using the optimized router
        # The router will handle its own dtype conversion
        routing_weights, expert_indices, router_aux_loss = self.router(x)
        
        # Update token processing count
        self.total_tokens_processed += batch_size * seq_len
        
        # Increment expert activation count
        expert_indices_flat = expert_indices.flatten()
        for expert_idx in range(self.num_experts):
            # Count occurrences of this expert in the indices
            count = torch.sum(expert_indices_flat == expert_idx).item()
            self.expert_activation_count[expert_idx] += count
            
        # Prepare empty tensor for expert outputs using the compute dtype for intermediate calculations
        expert_outputs = torch.zeros(
            batch_size, seq_len, self.output_dim, 
            device=device, dtype=compute_dtype
        )
        
        # Process each expert - batched processing for efficiency
        for expert_idx in range(self.num_experts):
            # Find positions where this expert is used
            expert_positions = []
            
            for k in range(self.top_k):
                batch_indices, seq_indices = torch.where(expert_indices[:, :, k] == expert_idx)
                if batch_indices.size(0) > 0:
                    # Store (batch_idx, seq_idx, k) for each occurrence
                    for b, s in zip(batch_indices, seq_indices):
                        expert_positions.append((b.item(), s.item(), k))
            
            # Skip if this expert is not used
            if not expert_positions:
                continue
                
            # Create input batch for this expert
            batch_idxs = [pos[0] for pos in expert_positions]
            seq_idxs = [pos[1] for pos in expert_positions]
            k_idxs = [pos[2] for pos in expert_positions]
            
            # Gather inputs for this expert
            expert_inputs = x[batch_idxs, seq_idxs]
            
            # Process the batch through this expert (precision handled inside)
            processed = self.experts[expert_idx](expert_inputs)
            
            # Get weights for this expert based on routing
            expert_weights = routing_weights[batch_idxs, seq_idxs, k_idxs]
            
            # Apply weights - scale outputs by routing weights
            processed = processed * expert_weights.unsqueeze(-1)
            
            # Accumulate to final output
            for i, (b, s, _) in enumerate(expert_positions):
                expert_outputs[b, s] += processed[i]
        
        # Convert output back to original input dtype if needed
        if expert_outputs.dtype != output_dtype:
            expert_outputs = expert_outputs.to(dtype=output_dtype)
            
        return expert_outputs, router_aux_loss
    
    def get_expert_activations(self) -> List[int]:
        """
        Get expert activation counts.
        
        Returns:
            List of integers with activation count for each expert
        """
        return self.expert_activation_count.copy()
    
    def get_expert_usage_stats(self) -> Dict[str, Any]:
        """
        Get detailed expert usage statistics.
        
        Returns:
            Dictionary with expert usage statistics
        """
        total = sum(self.expert_activation_count)
        if total == 0:
            return {"error": "No tokens processed yet"}
        
        percentages = [count / total * 100 for count in self.expert_activation_count]
        
        # Compute utilization metrics
        mean_util = sum(percentages) / len(percentages)
        nonzero_util = sum(1 for p in percentages if p > 0)
        utilization_rate = nonzero_util / self.num_experts * 100 if self.num_experts > 0 else 0
        
        # Get router statistics if available
        router_stats = {}
        if hasattr(self.router, 'get_router_statistics'):
            router_stats = self.router.get_router_statistics()
        
        return {
            "expert_activations": self.expert_activation_count,
            "expert_percentages": percentages,
            "total_tokens_processed": self.total_tokens_processed,
            "total_expert_activations": total,
            "mean_utilization": mean_util,
            "active_experts": nonzero_util,
            "utilization_rate": utilization_rate,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "routing_algorithm": getattr(self, 'routing_algorithm', 'top_k'),
            "router_stats": router_stats
        }


# Adapter for MoE block wrapper
class OptimizedMoEBlockWrapper(nn.Module):
    """
    Wrapper for decoder block that replaces MLP with optimized MoE.
    """
    def __init__(self, original_block, hidden_dim, mlp_dim, num_experts=8, top_k=2, 
                precision="fp32", routing_algorithm="top_k"):
        super().__init__()
        # Store the original block directly
        self.original_block = original_block
        self.use_moe = True
        
        # Create MoE layer with specified precision and routing algorithm
        self.moe = VectorizedMoELayer(
            input_dim=hidden_dim,
            hidden_dim=mlp_dim,
            output_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            precision=precision,
            routing_algorithm=routing_algorithm
        )
        
        # Store parameters for reference
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.precision = precision
        self.routing_algorithm = routing_algorithm
        
        print(f"Created optimized MoE block with {num_experts} experts and top-{top_k} routing "  
              f"using {precision} precision and {routing_algorithm} routing algorithm")
    
    def set_num_experts(self, num_experts):
        """Change the number of experts in the MoE layer."""
        self.num_experts = num_experts
        self.moe.change_num_experts(num_experts)
        
    def set_routing_algorithm(self, algorithm):
        """Set the routing algorithm for the MoE layer."""
        self.routing_algorithm = algorithm
        self.moe.set_routing_algorithm(algorithm)
    
    def forward(self, x, *args, **kwargs):
        """
        Forward pass through the MoE block.
        
        Args:
            x: Input tensor
            *args, **kwargs: Arguments to pass to the original block
            
        Returns:
            Output tensor
        """
        # Get intermediate representation after attention
        h = self.original_block(x, *args, **kwargs)
        
        # Check if MoE is enabled
        if not self.use_moe:
            return h
        
        # Process with MoE instead of MLP
        # The original implementation might return a tuple or just the tensor
        if isinstance(h, tuple):
            h_tensor, aux_data = h
        else:
            h_tensor = h
            aux_data = None
        
        # Run MoE
        moe_output, moe_aux_loss = self.moe(h_tensor)
        
        # Return in the same format as the original block
        if aux_data is not None:
            return moe_output, aux_data
        else:
            return moe_output
    
    # Cache handling methods
    def caches_are_enabled(self):
        """Check if caches are enabled in the original block."""
        return getattr(self.original_block, 'caches_are_enabled', lambda: False)()
    
    def setup_caches(self, batch_size, max_seq_len=None, dtype=None):
        """Set up caches in the original block."""
        if hasattr(self.original_block, 'setup_caches'):
            if max_seq_len is None:
                # Handle different signature versions
                self.original_block.setup_caches(batch_size, dtype)
            else:
                self.original_block.setup_caches(batch_size, max_seq_len, dtype)
    
    def reset_cache(self):
        """Reset cache in the original block."""
        if hasattr(self.original_block, 'reset_cache'):
            self.original_block.reset_cache()
            
    def reset_caches(self):
        """Reset caches in the original block (plural version)."""
        if hasattr(self.original_block, 'reset_caches'):
            self.original_block.reset_caches()
        elif hasattr(self.original_block, 'reset_cache'):
            self.original_block.reset_cache()

