#!/usr/bin/env python3
"""
Standalone test for the Llama 4 adapter
This script creates random tensors to verify the adapter works correctly
without requiring external model dependencies.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class Llama4Adapter(nn.Module):
    """
    Adapter to transform Llama 4's 5120-d hidden states to 
    match Sesame CSM's expected 4096-d input format.
    """
    def __init__(self, 
                 input_dim: int = 5120, 
                 output_dim: int = 4096, 
                 hidden_dim: Optional[int] = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim or output_dim
        
        # Two-layer MLP with GELU activation
        self.down_proj = nn.Linear(input_dim, output_dim)
        self.activation = nn.GELU()
        self.out_proj = nn.Linear(output_dim, output_dim)
        
        # Initialize weights for better convergence
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform Llama 4 hidden states to Sesame CSM compatible format.
        
        Args:
            x: Input tensor of shape [..., input_dim]
            
        Returns:
            Transformed tensor of shape [..., output_dim]
        """
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.out_proj(x)
        return x

def main():
    print("=== Testing Llama 4 Adapter with Synthetic Data ===")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create adapter
    print("Creating adapter model...")
    adapter = Llama4Adapter(input_dim=5120, output_dim=4096)
    adapter.to(device)
    print(f"Adapter created with input_dim={adapter.input_dim}, output_dim={adapter.output_dim}")
    
    try:
        # Test with random tensor (single vector)
        print("\nTest 1: Single hidden state vector")
        x = torch.randn(5120, device=device)
        y = adapter(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {y.shape}")
        assert y.shape == torch.Size([4096]), "Output shape mismatch"
        print("✅ Test 1 passed")
        
        # Test with batch of vectors
        print("\nTest 2: Batch of hidden states")
        batch = torch.randn(32, 5120, device=device)  # Batch of 32 hidden states
        batch_out = adapter(batch)
        print(f"Batch input shape: {batch.shape}")
        print(f"Batch output shape: {batch_out.shape}")
        assert batch_out.shape == torch.Size([32, 4096]), "Batch output shape mismatch"
        print("✅ Test 2 passed")
        
        # Test with sequence of hidden states
        print("\nTest 3: Sequence of hidden states (batch, seq_len, dim)")
        seq = torch.randn(16, 24, 5120, device=device)  # 16 examples, 24 tokens each
        seq_out = adapter(seq)
        print(f"Sequence input shape: {seq.shape}")
        print(f"Sequence output shape: {seq_out.shape}")
        assert seq_out.shape == torch.Size([16, 24, 4096]), "Sequence output shape mismatch"
        print("✅ Test 3 passed")
        
        # Save dummy model to verify serialization works
        print("\nTest 4: Serialization test")
        dummy_input = torch.randn(1, 5120, device=device)
        dummy_output = adapter(dummy_input)
        
        # Save model
        os.makedirs("checkpoints", exist_ok=True)
        save_path = "checkpoints/adapter_test.pt"
        torch.save(adapter.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        
        # Load model back and verify
        loaded_adapter = Llama4Adapter(input_dim=5120, output_dim=4096)
        loaded_adapter.load_state_dict(torch.load(save_path))
        loaded_adapter.to(device)
        loaded_output = loaded_adapter(dummy_input)
        
        # Verify that outputs are the same
        torch.testing.assert_close(dummy_output, loaded_output)
        print("✅ Test 4 passed: Model successfully saved and loaded")
        
        # Calculate model size
        param_size = sum(p.numel() * p.element_size() for p in adapter.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in adapter.buffers())
        total_size = param_size + buffer_size
        total_params = sum(p.numel() for p in adapter.parameters())
        
        print(f"\nAdapter Model Stats:")
        print(f"- Parameter count: {total_params:,}")
        print(f"- Model size: {total_size / (1024 * 1024):.2f} MB")
        print(f"- Estimated memory in FP32: {total_params * 4 / (1024 * 1024):.2f} MB")
        
        print("\n=== All tests passed! ===")
        print("The Llama 4 adapter successfully transforms from 5120d to 4096d")
        print("This adapter is ready to be used with the Llama 4 model to transform its hidden states")
        print("for compatibility with the Sesame CSM model.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    return True

if __name__ == "__main__":
    main()
