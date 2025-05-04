#!/usr/bin/env python3
"""
Test script to verify the data generation and training pipeline.
This creates a small synthetic dataset and trains the adapter for a few steps
to ensure everything is working correctly.
"""
import os
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llama4_adapter import Llama4Adapter

def generate_test_data(n_samples=100, llama4_dim=5120, csm_dim=4096, seed=42):
    """Generate a small synthetic dataset for testing"""
    print(f"Generating {n_samples} test samples...")
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    # Create a deterministic mapping matrix
    mapping = torch.randn(llama4_dim, csm_dim)
    mapping = mapping / torch.norm(mapping, dim=0, keepdim=True)
    
    # Generate input data
    llama4_states = torch.randn(n_samples, llama4_dim)
    
    # Generate target data with the mapping plus some noise
    csm_inputs = torch.matmul(llama4_states, mapping)
    csm_inputs = csm_inputs + torch.randn_like(csm_inputs) * 0.1
    
    return llama4_states, csm_inputs

def train_test(llama4_states, csm_inputs, epochs=5, batch_size=16, lr=1e-3):
    """Test training loop for the adapter"""
    print("Testing adapter training...")
    
    # Create dataset and dataloader
    dataset = TensorDataset(llama4_states, csm_inputs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = Llama4Adapter(input_dim=llama4_states.shape[1], output_dim=csm_inputs.shape[1])
    model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        total_cos_sim = 0.0
        batches = 0
        
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            y_pred = model(x)
            
            # Compute loss
            mse_loss = torch.nn.functional.mse_loss(y_pred, y)
            cos_loss = 1.0 - torch.nn.functional.cosine_similarity(y_pred, y, dim=1).mean()
            loss = 0.5 * mse_loss + 0.5 * cos_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_cos_sim += torch.nn.functional.cosine_similarity(y_pred, y, dim=1).mean().item()
            batches += 1
        
        # Print epoch metrics
        avg_loss = total_loss / batches
        avg_cos_sim = total_cos_sim / batches
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Cosine Similarity: {avg_cos_sim:.4f}")
    
    # Test the model on a separate batch
    test_x = torch.randn(10, llama4_states.shape[1], device=device)
    test_y = model(test_x)
    
    print(f"\nTest output shape: {test_y.shape}")
    print("Training test completed successfully!")
    
    return model

def save_test_data(llama4_states, csm_inputs, output_path="data/test_data.pt"):
    """Save the test data to a file"""
    print(f"Saving test data to {output_path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save data
    torch.save({
        "llama4_states": llama4_states,
        "csm_inputs": csm_inputs,
        "prompts": ["Test prompt"] * len(llama4_states),
        "method": "test_synthetic"
    }, output_path)
    
    print(f"Test data saved successfully to {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Test the data generation and training pipeline")
    parser.add_argument("--n_samples", type=int, default=100, 
                       help="Number of test samples to generate")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--save_data", action="store_true",
                       help="Save the generated test data")
    parser.add_argument("--output", type=str, default="data/test_data.pt",
                       help="Output path for test data")
    parser.add_argument("--save_model", action="store_true",
                       help="Save the trained test model")
    parser.add_argument("--model_output", type=str, default="checkpoints/test_adapter.pt",
                       help="Output path for test model")
    args = parser.parse_args()
    
    # Generate test data
    llama4_states, csm_inputs = generate_test_data(n_samples=args.n_samples)
    
    # Save test data if requested
    if args.save_data:
        save_test_data(llama4_states, csm_inputs, args.output)
    
    # Train the model
    model = train_test(llama4_states, csm_inputs, epochs=args.epochs)
    
    # Save the model if requested
    if args.save_model:
        os.makedirs(os.path.dirname(args.model_output), exist_ok=True)
        torch.save(model.state_dict(), args.model_output)
        print(f"Test model saved to {args.model_output}")

if __name__ == "__main__":
    main()
