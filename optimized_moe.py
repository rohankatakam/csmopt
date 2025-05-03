#!/usr/bin/env python3
"""
Optimized MoE implementation with vectorized operations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import List, Tuple, Optional, Dict, Any

class VectorizedMoEGating(nn.Module):
    """
    Vectorized implementation of top-k gating for Mixture of Experts.
    """
    def __init__(self, input_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(input_dim, num_experts, bias=False)
        
        # Initialize router weights 
        nn.init.normal_(self.router.weight, mean=0.0, std=0.1)
    
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
        
        # Compute router logits - [batch_size, seq_len, num_experts]
        batch_size, seq_len, _ = x.shape
        
        # Flatten input for efficient routing
        x_flat = x.reshape(-1, self.input_dim)
        
        # Get router logits
        router_logits = self.router(x_flat)  # [batch_size*seq_len, num_experts]
        
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
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.up_proj = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.down_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the expert.
        
        Args:
            x: Input tensor of any shape as long as last dim is input_dim
            
        Returns:
            Output tensor with same shape as input except last dim is output_dim
        """
        return self.down_proj(self.act(self.up_proj(x)))


class VectorizedMoELayer(nn.Module):
    """
    Vectorized MoE layer with improved batch processing.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, 
                 num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Create gating network
        self.gate = VectorizedMoEGating(input_dim, num_experts, top_k)
        
        # Create experts
        self.experts = nn.ModuleList([
            VectorizedMoEExpertLayer(input_dim, hidden_dim, output_dim)
            for _ in range(num_experts)
        ])
        
        # Initialize tracking for expert usage
        self.expert_activation_count = [0] * num_experts
        self.total_tokens_processed = 0
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Forward pass through the MoE layer with vectorized operations.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Tuple of (output, aux_loss)
            - output: Tensor of shape [batch_size, seq_len, output_dim]
            - aux_loss: Auxiliary load balancing loss (0.0 for now)
        """
        batch_size, seq_len, embed_dim = x.shape
        device = x.device
        
        # Reshape for efficient processing
        # We'll process all token embeddings as a single batch
        x_flat = x.reshape(-1, embed_dim)  # [batch_size*seq_len, embed_dim]
        
        # Get routing weights and indices
        routing_weights_orig, expert_indices_orig = self.gate(x)
        
        # Reshape to match x_flat
        routing_weights = routing_weights_orig.reshape(-1, self.top_k)  # [batch_size*seq_len, top_k]
        expert_indices = expert_indices_orig.reshape(-1, self.top_k)    # [batch_size*seq_len, top_k]
        
        # Count tokens for tracking expert usage
        flat_indices = expert_indices.flatten()
        for expert_idx in range(self.num_experts):
            self.expert_activation_count[expert_idx] += (flat_indices == expert_idx).sum().item()
        self.total_tokens_processed += batch_size * seq_len * self.top_k
        
        # Initialize output tensor
        final_output = torch.zeros_like(x_flat)  # [batch_size*seq_len, embed_dim]
        
        # Process each expert in parallel
        # This approach processes all tokens for an expert at once
        for expert_idx in range(self.num_experts):
            # For each token position [0...batch_size*seq_len-1]
            # Find all positions where this expert_idx appears in any top-k position
            # Check if any token uses this expert
            expert_mask = torch.any(expert_indices == expert_idx, dim=1)  # [batch_size*seq_len]
            
            if not expert_mask.any():
                continue  # Skip if no tokens use this expert
            
            # Get all tokens that use this expert
            selected_tokens = x_flat[expert_mask]  # [num_tokens, embed_dim]
            
            # Process all these tokens at once
            expert_output = self.experts[expert_idx](selected_tokens)  # [num_tokens, embed_dim]
            
            # For each token, find corresponding weight
            # First, get positions in the flattened array
            token_positions = expert_mask.nonzero(as_tuple=True)[0]  # [num_tokens]
            
            # For each selected token, find which position (0...top_k-1) has this expert
            token_routing_weights = torch.zeros(token_positions.size(0), device=device)
            
            for k in range(self.top_k):
                # Create mask for tokens where this expert is at position k
                k_mask = expert_indices[token_positions, k] == expert_idx
                if k_mask.any():
                    # Get the weights for these tokens
                    token_routing_weights[k_mask] = routing_weights[token_positions[k_mask], k]
            
            # Apply weights to expert outputs
            weighted_output = expert_output * token_routing_weights.unsqueeze(1)
            
            # Add to final output
            final_output[token_positions] += weighted_output
            
        # Reshape back to original dimensions
        final_output = final_output.reshape(batch_size, seq_len, embed_dim)
        
        return final_output, 0.0  # No auxiliary loss for now
    
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
        
        return {
            "expert_activations": self.expert_activation_count,
            "expert_percentages": percentages,
            "total_tokens_processed": self.total_tokens_processed,
            "total_expert_activations": total
        }


# Adapter for MoE block wrapper
class OptimizedMoEBlockWrapper(nn.Module):
    """
    Wrapper for decoder block that replaces MLP with optimized MoE.
    """
    def __init__(self, original_block, hidden_dim, mlp_dim, num_experts=8, top_k=2):
        super().__init__()
        self._original_block = original_block  # Use _original_block as internal attribute name
        self.use_moe = True
        
        # Create MoE layer
        self.moe = VectorizedMoELayer(
            input_dim=hidden_dim,
            hidden_dim=mlp_dim,
            output_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k
        )
        
        # Store parameters for reference
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        print(f"Created optimized MoE block with {num_experts} experts and top-{top_k} routing")
    
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
        h = self._original_block(x, *args, **kwargs)
        
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
    
    # Required methods for cache handling
    def caches_are_enabled(self):
        """Check if caches are enabled in the original block."""
        if hasattr(self._original_block, 'caches_are_enabled'):
            return self._original_block.caches_are_enabled()
        return False
    
    def setup_caches(self, batch_size, max_seq_len, dtype=None):
        """Set up caches in the original block."""
        if hasattr(self._original_block, 'setup_caches'):
            self._original_block.setup_caches(batch_size, max_seq_len, dtype)
    
    def reset_cache(self):
        """Reset cache in the original block."""
        if hasattr(self._original_block, 'reset_cache'):
            self._original_block.reset_cache()
    
    def __getattr__(self, name):
        """Delegate attribute access to the original block."""
        if name != '_original_block' and hasattr(self._original_block, name):
            return getattr(self._original_block, name)
        raise AttributeError(f"{self.__class__.__name__} has no attribute '{name}'")
