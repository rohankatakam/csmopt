"""
MoE (Mixture of Experts) routing mechanism for CSM.
Inspired by LLaMA 4's sparse MoE architecture.

This module implements a lightweight adaptive routing mechanism that
dynamically activates only the most relevant expert layers in CSM's decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEGating(nn.Module):
    """
    Top-k gating module for Mixture of Experts.
    """
    def __init__(self, 
                 input_dim: int, 
                 num_experts: int, 
                 k: int = 2, 
                 capacity_factor: float = 1.0,
                 normalize_gates: bool = True,
                 noise_epsilon: float = 1e-2):
        """
        Initialize the MoE gating module.
        
        Args:
            input_dim: Input dimension
            num_experts: Number of experts
            k: Number of experts to route to (default: 2 for top-2 routing)
            capacity_factor: Capacity factor to handle load balancing
            normalize_gates: Whether to normalize gate values
            noise_epsilon: Noise epsilon for training
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.k = min(k, num_experts)  # k can't be larger than num_experts
        self.capacity_factor = capacity_factor
        self.normalize_gates = normalize_gates
        self.noise_epsilon = noise_epsilon
        
        # Gate linear layer to compute routing logits
        self.gate = nn.Linear(input_dim, num_experts, bias=False)
        
        # Initialize gate weights
        with torch.no_grad():
            # Use orthogonal initialization for better routing
            nn.init.orthogonal_(self.gate.weight, gain=0.1)

    def forward(self, x):
        """
        Compute routing probabilities and expert indices.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            router_logits: Raw routing logits
            routing_weights: Normalized weights for selected experts
            expert_indices: Selected expert indices for each token
            combine_weights: Weights for combining expert outputs
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype
        
        # Ensure gate weights are on the same device as input
        if self.gate.weight.device != device:
            self.gate.to(device=device, dtype=dtype)
        
        # Compute gate logits
        router_logits = self.gate(x)  # [batch_size, seq_len, num_experts]
        
        # Add noise during training for exploration (not during inference)
        if self.training and self.noise_epsilon > 0:
            router_logits += torch.randn_like(router_logits) * self.noise_epsilon
        
        # Find top-k experts
        routing_weights, expert_indices = torch.topk(router_logits, k=self.k, dim=-1)
        
        # Convert to probabilities with softmax
        if self.normalize_gates:
            routing_weights = F.softmax(routing_weights, dim=-1)
        
        # Reshape for the router output
        # expert_indices shape: [batch_size, seq_len, k]
        # routing_weights shape: [batch_size, seq_len, k]
        
        # No need to create combine_weights separately, just return routing_weights
        # This simplifies the implementation and avoids any dimension mismatch
        
        return router_logits, routing_weights, expert_indices, routing_weights
        
    def compute_router_z_loss(self, router_logits):
        """
        Compute router z-loss to encourage balanced expert routing.
        
        Based on paper: https://arxiv.org/abs/2101.03961
        """
        # Mean over batch and sequence dimensions
        mean_router_logits = router_logits.mean(dim=(0, 1))
        # Z-loss encourages router logits to be close to zero
        router_z_loss = torch.mean(torch.square(torch.logsumexp(router_logits, dim=-1)))
        return router_z_loss


class MoEExpertLayer(nn.Module):
    """
    Expert layer for Mixture of Experts.
    Each expert is a simple MLP with two linear layers and activation.
    """
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int,
                 output_dim: int,
                 activation: nn.Module = nn.GELU()):
        super().__init__()
        self.up_proj = nn.Linear(input_dim, hidden_dim)
        self.activation = activation
        self.down_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Make sure model is on same device as input
        device = x.device
        dtype = x.dtype
        
        if self.up_proj.weight.device != device:
            self.to(device=device, dtype=dtype)
            
        x = self.up_proj(x)
        x = self.activation(x)
        x = self.down_proj(x)
        return x


class MoELayer(nn.Module):
    """
    Mixture of Experts layer with top-k routing.
    """
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int,
                 output_dim: int,
                 num_experts: int = 8, 
                 k: int = 2,
                 capacity_factor: float = 1.0,
                 router_z_loss_coef: float = 0.001):
        """
        Initialize the MoE layer.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension of experts
            output_dim: Output dimension
            num_experts: Number of experts
            k: Number of experts to route to
            capacity_factor: Capacity factor for load balancing
            router_z_loss_coef: Coefficient for router z-loss
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.k = k
        self.router_z_loss_coef = router_z_loss_coef
        
        # Initialize the router
        self.router = MoEGating(
            input_dim=input_dim, 
            num_experts=num_experts, 
            k=k, 
            capacity_factor=capacity_factor
        )
        
        # Initialize experts
        self.experts = nn.ModuleList([
            MoEExpertLayer(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim
            ) for _ in range(num_experts)
        ])
        
        # Initialize expert activation tracking
        self.expert_activation_count = torch.zeros(num_experts)
        
    def reset_activation_counts(self):
        """Reset expert activation counts."""
        self.expert_activation_count = torch.zeros(self.num_experts)
        
    def _update_activation_counts(self, expert_indices):
        """Update expert activation counts during forward pass."""
        # Count activations on CPU to avoid synchronization issues
        for expert_idx in expert_indices.cpu().view(-1):
            self.expert_activation_count[expert_idx.item()] += 1
    
    def forward(self, x):
        """
        Forward pass through the MoE layer.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            output: Output tensor of shape [batch_size, seq_len, output_dim]
            aux_loss: Auxiliary loss for router
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype
        
        # Move the entire module to the input's device if needed
        if next(self.parameters()).device != device:
            self.to(device=device, dtype=dtype)
        
        # Get router outputs
        router_logits, routing_weights, expert_indices, combine_weights = self.router(x)
        
        # Update activation counts
        self._update_activation_counts(expert_indices)
        
        # Compute router z-loss
        router_z_loss = self.router.compute_router_z_loss(router_logits)
        aux_loss = self.router_z_loss_coef * router_z_loss
        
        # Dispatch to experts (simple implementation)
        final_output = torch.zeros(batch_size, seq_len, self.output_dim, device=device, dtype=dtype)
        
        # Process each token through its assigned experts
        with torch.amp.autocast(device_type='cuda', enabled=x.dtype == torch.bfloat16):
            for batch_idx in range(batch_size):
                for seq_idx in range(seq_len):
                    for k_idx in range(self.k):
                        expert_idx = expert_indices[batch_idx, seq_idx, k_idx].item()
                        weight = routing_weights[batch_idx, seq_idx, k_idx].item()
                        
                        # Get expert output for this token
                        token_input = x[batch_idx, seq_idx].unsqueeze(0)  # [1, input_dim]
                        expert_output = self.experts[expert_idx](token_input)  # [1, output_dim]
                        
                        # Combine weighted expert output
                        final_output[batch_idx, seq_idx] += weight * expert_output.squeeze(0)
        
        return final_output, aux_loss
    
    def get_expert_activations(self):
        """Get the expert activation counts for analytics without requiring input."""
        return self.expert_activation_count.clone()


class MoEBlockWrapper(nn.Module):
    """
    Wrapper for a decoder block to add MoE capabilities.
    """
    def __init__(self, 
                 original_block: nn.Module,
                 embed_dim: int,
                 mlp_dim: int,
                 num_experts: int = 8,
                 k: int = 2):
        """
        Initialize the MoE block wrapper.
        
        Args:
            original_block: Original decoder block to wrap
            embed_dim: Embedding dimension
            mlp_dim: MLP dimension
            num_experts: Number of experts
            k: Number of experts to route to
        """
        super().__init__()
        self.original_block = original_block
        
        # Store KV cache attributes
        self._kv_cache_enabled = False
        self._kv_cache = None
        
        # Replace the MLP with MoE
        self.moe_layer = MoELayer(
            input_dim=embed_dim,
            hidden_dim=mlp_dim,
            output_dim=embed_dim,
            num_experts=num_experts,
            k=k
        )
        
        # Flag to enable/disable MoE
        self.use_moe = True
        
    def forward(self, x, *args, **kwargs):
        """
        Forward pass through the MoE block.
        
        Args:
            x: Input tensor
            *args, **kwargs: Arguments to pass to the original block
            
        Returns:
            output: Output tensor (note: we store aux_loss as an attribute but don't return it)
        """
        # Pass through attention and first part of the block
        h = self.original_block(x, *args, **kwargs)
        
        # If MoE is enabled, replace the MLP part with MoE routing
        if self.use_moe:
            moe_output, aux_loss = self.moe_layer(h)
            # Store aux_loss as an attribute instead of returning it
            self.last_aux_loss = aux_loss.item() if hasattr(aux_loss, 'item') else aux_loss
            return moe_output
        else:
            # Use the original output
            self.last_aux_loss = 0.0
            return h
    
    # Implement the cache-related methods directly
    def caches_are_enabled(self):
        """Check if KV cache is enabled."""
        # If our MoEBlockWrapper is being called directly for this check
        return hasattr(self.original_block, 'caches_are_enabled') and self.original_block.caches_are_enabled()
    
    def setup_caches(self, batch_size, dtype, max_seq_len=None, device=None):
        """Setup the KV cache for this block."""
        if hasattr(self.original_block, 'setup_caches'):
            self.original_block.setup_caches(batch_size, dtype, max_seq_len, device)
    
    def reset_caches(self):
        """Reset the KV cache for this block."""
        if hasattr(self.original_block, 'reset_caches'):
            self.original_block.reset_caches()
    
    def reset_cache(self):
        """Reset the KV cache for this block (singular version required by torchtune)."""
        if hasattr(self.original_block, 'reset_cache'):
            self.original_block.reset_cache()


# Utility for patching the CSM decoder with MoE blocks
def patch_llama_with_moe(model, block_indices, embed_dim, mlp_dim, num_experts=8, k=2):
    """
    Patch LLaMA decoder blocks with MoE routing.
    
    Args:
        model: The CSM model to patch
        block_indices: Indices of blocks to patch with MoE
        embed_dim: Embedding dimension
        mlp_dim: MLP dimension
        num_experts: Number of experts
        k: Number of experts to route to
        
    Returns:
        patched_model: Model with MoE blocks
        moe_blocks: Dictionary of MoE blocks for monitoring
    """
    moe_blocks = {}
    
    # Get decoder layers
    decoder = model.decoder
    
    # Wrap specified blocks with MoE
    for idx in block_indices:
        # Check if the index is valid
        if hasattr(decoder, 'layers') and idx < len(decoder.layers):
            original_block = decoder.layers[idx]
            moe_block = MoEBlockWrapper(
                original_block=original_block,
                embed_dim=embed_dim,
                mlp_dim=mlp_dim,
                num_experts=num_experts,
                k=k
            )
            
            # Replace the original block
            decoder.layers[idx] = moe_block
            moe_blocks[f"decoder_block_{idx}"] = moe_block
    
    return model, moe_blocks
