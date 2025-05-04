"""
Optimized MoE routing algorithms for CSM.

This module implements advanced routing strategies for MoE to improve expert utilization
and load balancing. It includes:
1. Balanced top-k routing (based on Switch Transformers)
2. Expert-choice routing (S-BASE routing)
3. Token-choice routing with load balancing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Optional, Any


class OptimizedMoERouter(nn.Module):
    """
    Optimized router for Mixture of Experts with multiple routing strategies.
    """
    def __init__(
        self, 
        input_dim: int, 
        num_experts: int, 
        top_k: int = 2,
        routing_algorithm: str = "top_k",
        capacity_factor: float = 1.5,
        aux_loss_weight: float = 0.01
    ):
        """
        Initialize the MoE router with configurable algorithm.
        
        Args:
            input_dim: Input dimension
            num_experts: Number of experts
            top_k: Number of experts to route to
            routing_algorithm: Routing algorithm to use:
                - "top_k": Standard top-k routing
                - "balanced": Top-k with load balancing
                - "expert_choice": Experts choose tokens
            capacity_factor: Overallocation factor for balanced routing
            aux_loss_weight: Weight for auxiliary load balancing loss
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.routing_algorithm = routing_algorithm
        self.capacity_factor = capacity_factor
        self.aux_loss_weight = aux_loss_weight
        
        # Router network (maps tokens to experts)
        self.router = nn.Linear(input_dim, num_experts, bias=False)
        
        # Initialize with scaled normal distribution
        nn.init.normal_(self.router.weight, mean=0.0, std=0.1 * (input_dim ** -0.5))
        
        # Keep track of tokens dropped due to load balancing
        self.dropped_tokens = 0
        self.total_tokens = 0
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts using the configured routing algorithm.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Tuple of (routing_weights, expert_indices, aux_loss)
            - routing_weights: Tensor of shape [batch_size, seq_len, top_k]
            - expert_indices: Tensor of shape [batch_size, seq_len, top_k]
            - aux_loss: Auxiliary load balancing loss
        """
        batch_size, seq_len, _ = x.shape
        tokens_per_batch = batch_size * seq_len
        self.total_tokens += tokens_per_batch
        
        # Ensure router is on the same device and dtype as input
        device = x.device
        dtype = x.dtype
        if self.router.weight.device != device or self.router.weight.dtype != dtype:
            self.router.to(device=device, dtype=dtype)
            
        # Compute router logits
        router_logits = self.router(x)  # [batch_size, seq_len, num_experts]
        
        # Route according to the chosen algorithm
        if self.routing_algorithm == "balanced":
            return self._balanced_routing(router_logits, batch_size, seq_len)
        elif self.routing_algorithm == "expert_choice":
            return self._expert_choice_routing(router_logits, batch_size, seq_len)
        else:  # Default to standard top-k
            return self._standard_top_k_routing(router_logits, batch_size, seq_len)
    
    def _standard_top_k_routing(
        self, 
        router_logits: torch.Tensor,
        batch_size: int, 
        seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Standard top-k routing algorithm.
        
        Args:
            router_logits: Router logits of shape [batch_size, seq_len, num_experts]
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Tuple of (routing_weights, expert_indices, aux_loss)
        """
        # Get top-k experts per token (standard routing)
        router_probs = F.softmax(router_logits, dim=-1)
        routing_weights, expert_indices = torch.topk(
            router_probs, k=self.top_k, dim=-1
        )
        
        # Calculate load balancing loss - penalize highly imbalanced usage
        # We want router_probs to be close to uniform (1/num_experts)
        router_prob_mean = router_probs.mean(dim=[0, 1])
        aux_loss = self.aux_loss_weight * (
            self.num_experts * (router_prob_mean * router_prob_mean).sum() - 1.0
        )
        
        return routing_weights, expert_indices, aux_loss
    
    def _balanced_routing(
        self, 
        router_logits: torch.Tensor,
        batch_size: int, 
        seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load-balanced routing algorithm based on Switch Transformers.
        
        Args:
            router_logits: Router logits of shape [batch_size, seq_len, num_experts]
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Tuple of (routing_weights, expert_indices, aux_loss)
        """
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Compute expert capacity - how many tokens each expert can process
        # With balanced routing, we slightly over-allocate capacity
        capacity = int(self.capacity_factor * batch_size * seq_len * self.top_k / self.num_experts)
        
        # Get top-k routing probabilities and corresponding expert indices
        routing_weights, expert_indices = torch.topk(
            router_probs, k=self.top_k, dim=-1
        )
        
        # Create mask for balancing - shape [batch_size, seq_len, top_k]
        mask = torch.ones_like(routing_weights)
        
        # Safe version of balanced routing with explicit bounds checking
        for expert_idx in range(self.num_experts):
            for k in range(self.top_k):
                # Count tokens routed to this expert at position k
                expert_mask = (expert_indices[:, :, k] == expert_idx)
                token_count = expert_mask.sum().item()
                
                if token_count > capacity:
                    # Get priorities for this expert based on router_probs
                    priorities = router_probs[:, :, expert_idx].clone()
                    # Zero out priorities for tokens not assigned to this expert at position k
                    priorities = priorities * expert_mask.float()
                    
                    # Flatten for easier processing
                    flat_priorities = priorities.reshape(-1)
                    flat_mask = expert_mask.reshape(-1)
                    
                    # Get indices of tokens assigned to this expert
                    assigned_indices = torch.nonzero(flat_mask).squeeze(-1)
                    
                    # If we have valid indices, sort by priority and drop lowest
                    if assigned_indices.numel() > 0:
                        # Sort by priority in descending order
                        sorted_indices = assigned_indices[torch.argsort(flat_priorities[assigned_indices], descending=True)]
                        
                        # Get indices to drop (lowest priority ones beyond capacity)
                        if len(sorted_indices) > capacity:
                            to_drop = sorted_indices[capacity:]
                            # Update the mask for this specific k position
                            mask_flat_view = mask[:, :, k].reshape(-1)
                            mask_flat_view[to_drop] = 0.0
                            self.dropped_tokens += len(to_drop)
        
        # Apply the mask and renormalize weights
        routing_weights = routing_weights * mask
        routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Calculate load balancing loss 
        expert_counts = torch.zeros(self.num_experts, device=router_logits.device)
        for i in range(self.num_experts):
            expert_counts[i] = (expert_indices == i).sum().float()
        aux_loss = self.aux_loss_weight * torch.var(expert_counts) / (torch.mean(expert_counts) ** 2 + 1e-8)
        
        return routing_weights, expert_indices, aux_loss
    
    def _expert_choice_routing(
        self, 
        router_logits: torch.Tensor,
        batch_size: int, 
        seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Expert-choice routing algorithm.
        Experts choose tokens they want to process rather than tokens choosing experts.
        
        Args:
            router_logits: Router logits of shape [batch_size, seq_len, num_experts]
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Tuple of (routing_weights, expert_indices, aux_loss)
        """
        device = router_logits.device
        # Get normalized router logits (softmax over experts for each token)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Transpose to [num_experts, batch_size * seq_len]
        flattened_logits = router_logits.reshape(batch_size * seq_len, self.num_experts).t()
        
        # Each expert selects top-k tokens (with a safety limit)
        max_tokens = batch_size * seq_len
        tokens_per_expert = min(
            int(self.capacity_factor * batch_size * seq_len * self.top_k / self.num_experts),
            max_tokens  # This ensures we don't try to select more tokens than exist
        )
        
        # Get the top-k tokens per expert (with safety checks)
        k_tokens = min(tokens_per_expert, max_tokens)
        if k_tokens <= 0:
            k_tokens = 1  # Ensure we select at least one token
        
        # Create a simpler and more robust expert_choice routing approach
        # First, create our output tensors
        routing_weights = torch.zeros(
            batch_size, seq_len, self.top_k, device=device, dtype=router_logits.dtype
        )
        expert_assignments = torch.zeros(
            batch_size, seq_len, self.top_k, device=device, dtype=torch.long
        )
        
        # Track how many experts are assigned to each token
        token_expert_count = torch.zeros(batch_size * seq_len, device=device, dtype=torch.int32)
        
        # Reshape router_logits to [num_experts, batch_size, seq_len]
        reshaped_logits = router_logits.permute(2, 0, 1)
        
        # For each expert, determine which tokens it wants to process
        for expert_idx in range(self.num_experts):
            # Get this expert's logits for all tokens
            expert_logits = reshaped_logits[expert_idx].reshape(-1)  # Flatten to [batch_size * seq_len]
            
            # Calculate tokens per expert (with safety margin)
            safe_k = min(k_tokens, batch_size * seq_len)
            
            # If there are no tokens to process, skip this expert
            if safe_k <= 0:
                continue
                
            # Select top tokens for this expert
            top_scores, top_indices = torch.topk(expert_logits, k=safe_k)
            
            # For each selected token
            for token_idx, score in zip(top_indices, top_scores):
                # Skip invalid indices
                if token_idx >= batch_size * seq_len:
                    continue
                    
                # Convert to batch, seq indices
                b = (token_idx // seq_len).item()
                s = (token_idx % seq_len).item()
                
                # Get current count of experts for this token
                count = token_expert_count[token_idx].item()
                
                # Only assign if token hasn't been assigned to max experts yet
                if count < self.top_k:
                    # Assign this expert to the token
                    expert_assignments[b, s, count] = expert_idx
                    routing_weights[b, s, count] = score.item()
                    token_expert_count[token_idx] += 1
                    
        # Normalize routing weights to sum to 1
        weight_sum = routing_weights.sum(dim=-1, keepdim=True)
        # Avoid division by zero
        mask = (weight_sum > 0).float()
        normalized_weights = routing_weights / (weight_sum + 1e-10) * mask
        
        # Create load balancing loss
        # Count assignments per expert
        expert_counts = torch.zeros(self.num_experts, device=device)
        for expert_idx in range(self.num_experts):
            expert_counts[expert_idx] = (expert_assignments == expert_idx).sum().float()
        
        # Compute coefficient of variation as our auxiliary loss
        # We want experts to be used uniformly
        mean_count = expert_counts.mean()
        if mean_count > 0:
            aux_loss = self.aux_loss_weight * (expert_counts.var() / (mean_count ** 2 + 1e-5))
        else:
            aux_loss = torch.tensor(0.0, device=device)
            
        return normalized_weights, expert_assignments, aux_loss
    
    def set_routing_algorithm(self, algorithm: str) -> None:
        """
        Set the routing algorithm.
        
        Args:
            algorithm: The routing algorithm to use.
        """
        valid_algorithms = ["top_k", "balanced", "expert_choice"]
        if algorithm not in valid_algorithms:
            print(f"Warning: Invalid routing algorithm '{algorithm}'. Using 'top_k' instead.")
            algorithm = "top_k"
        
        self.routing_algorithm = algorithm
        print(f"Router set to '{algorithm}' routing algorithm")
    
    def get_router_statistics(self) -> Dict[str, Any]:
        """
        Get router statistics.
        
        Returns:
            Dictionary containing router statistics.
        """
        drop_rate = 0.0
        if self.total_tokens > 0:
            drop_rate = float(self.dropped_tokens) / float(self.total_tokens)
            
        return {
            "routing_algorithm": self.routing_algorithm,
            "total_tokens_processed": self.total_tokens,
            "dropped_tokens": self.dropped_tokens,
            "token_drop_rate": drop_rate,
            "num_experts": self.num_experts,
            "top_k": self.top_k
        }
