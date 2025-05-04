#!/usr/bin/env python3
"""
Generate synthetic paired data for training the Llama 4 adapter.
This script provides three options:
1. Fully synthetic data (random tensors)
2. Real Llama 4 embeddings paired with synthetic CSM inputs
3. Transformation-based synthetic data (to maintain semantic relationships)

This avoids dependency issues with the CSM model while allowing
training to proceed.
"""
import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_fully_synthetic_data(
    n_samples: int = 5000,
    llama4_dim: int = 5120,
    csm_dim: int = 4096,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate fully synthetic data pairs.
    
    This creates random tensors as stand-ins for Llama 4 hidden states and CSM inputs.
    The relationship between them is also random, so this is only useful for
    testing the training pipeline, not for producing a useful adapter.
    
    Args:
        n_samples: Number of samples to generate
        llama4_dim: Dimension of Llama 4 hidden states
        csm_dim: Dimension of CSM inputs
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (llama4_states, csm_inputs)
    """
    print(f"Generating {n_samples} fully synthetic data pairs...")
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate random tensors
    llama4_states = torch.randn(n_samples, llama4_dim)
    csm_inputs = torch.randn(n_samples, csm_dim)
    
    return llama4_states, csm_inputs

def get_prompt_texts(n: int = 5000) -> List[str]:
    """
    Generate prompt texts for data collection.
    
    Args:
        n: Number of prompts to generate
        
    Returns:
        List of prompt texts
    """
    # Simple prompt templates
    templates = [
        "Hey, how are you doing today?",
        "What's the weather like in your area?",
        "I've been thinking about learning to play the guitar. Any advice?",
        "Can you tell me about the history of artificial intelligence?",
        "What's your favorite book and why?",
        "How would you explain quantum physics to a child?",
        "What are some good exercises for staying fit at home?",
        "If you could travel anywhere in the world, where would you go?",
        "What's the best way to learn a new language?",
        "Tell me about the most interesting scientific discovery in the last decade.",
        "What are some healthy breakfast ideas?",
        "How does blockchain technology work?",
        "What's your favorite film of all time?",
        "How can I improve my public speaking skills?",
        "What are the benefits of meditation?",
        "Can you recommend some good podcasts?",
        "How do you stay motivated when working on difficult projects?",
        "What are some effective strategies for time management?",
        "How is artificial intelligence changing healthcare?",
        "What are some easy recipes for beginners?",
    ]
    
    # For a real implementation, we'd want many more diverse prompts
    # This is just a placeholder with enough variation for basic training
    
    # Ensure we have enough prompts by repeating if necessary
    all_prompts = templates * (n // len(templates) + 1)
    return all_prompts[:n]

def extract_llama4_embeddings(
    model_name: str,
    prompts: List[str],
    device: str = "cuda",
    batch_size: int = 4,
    use_4bit: bool = True
) -> torch.Tensor:
    """
    Extract real Llama 4 embeddings for a list of prompts.
    
    Args:
        model_name: Name of the Llama 4 model to use
        prompts: List of prompt texts
        device: Device to run the model on
        batch_size: Batch size for processing
        use_4bit: Whether to use 4-bit quantization
        
    Returns:
        Tensor of Llama 4 hidden states
    """
    print(f"Loading Llama 4 model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model
    print(f"Loading model with quantization: {use_4bit}")
    load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": device if device != "cpu" else None,
        "output_hidden_states": True
    }
    
    if use_4bit:
        load_kwargs["load_in_4bit"] = True
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to 8-bit quantization...")
        load_kwargs["load_in_8bit"] = True
        if "load_in_4bit" in load_kwargs:
            del load_kwargs["load_in_4bit"]
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        except Exception as e:
            print(f"Error loading model with 8-bit quantization: {e}")
            print("Falling back to CPU (this will be slow)...")
            load_kwargs["device_map"] = None
            if "load_in_8bit" in load_kwargs:
                del load_kwargs["load_in_8bit"]
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            device = "cpu"
    
    print(f"Model loaded successfully on {device}")
    
    # Extract hidden states
    hidden_states = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Extracting Llama 4 hidden states"):
        batch_texts = prompts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True)
        
        # Move inputs to the correct device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        
        with torch.no_grad():
            try:
                outputs = model(**inputs)
                
                # Get the hidden states from the last layer
                last_hidden_states = outputs.hidden_states[-1]
                
                # Extract hidden state for the last token of each sequence
                for j in range(len(batch_texts)):
                    seq_len = inputs["attention_mask"][j].sum().item()
                    hidden_state = last_hidden_states[j, seq_len - 1].cpu()
                    hidden_states.append(hidden_state)
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                # Generate random embeddings as fallback
                for j in range(len(batch_texts)):
                    hidden_states.append(torch.randn(model.config.hidden_size))
    
    return torch.stack(hidden_states)

def generate_transformation_based_data(
    llama4_states: torch.Tensor,
    csm_dim: int = 4096,
    noise_level: float = 0.1,
    seed: int = 42
) -> torch.Tensor:
    """
    Generate synthetic CSM inputs based on Llama 4 embeddings using a
    consistent transformation plus noise.
    
    This creates a deterministic relationship between inputs and outputs,
    which is more realistic than fully random data.
    
    Args:
        llama4_states: Tensor of Llama 4 hidden states
        csm_dim: Dimension of CSM inputs
        noise_level: Level of noise to add (0-1)
        seed: Random seed for reproducibility
        
    Returns:
        Tensor of synthetic CSM inputs
    """
    print(f"Generating transformation-based CSM inputs with noise level {noise_level}...")
    
    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get dimensions
    n_samples = llama4_states.shape[0]
    llama4_dim = llama4_states.shape[1]
    
    # Create a fixed random projection matrix
    projection = torch.randn(llama4_dim, csm_dim)
    
    # Normalize the projection matrix
    projection = projection / torch.norm(projection, dim=0, keepdim=True)
    
    # Apply projection
    csm_inputs = torch.matmul(llama4_states, projection)
    
    # Add noise
    if noise_level > 0:
        noise = torch.randn_like(csm_inputs) * noise_level
        csm_inputs = csm_inputs + noise
    
    # Normalize to have similar scale as real CSM inputs would have
    csm_inputs = csm_inputs / torch.std(csm_inputs) * 0.5
    
    return csm_inputs

def main():
    parser = argparse.ArgumentParser(description="Generate data for Llama 4 to CSM adapter training")
    parser.add_argument("--output", type=str, default="data/adapter_training_data.pt", 
                       help="Output file path")
    parser.add_argument("--n_samples", type=int, default=5000, 
                       help="Number of samples to generate")
    parser.add_argument("--llama4_model", type=str, 
                       default="meta-llama/Llama-4-Scout-17B-16E-Instruct", 
                       help="Llama 4 model name")
    parser.add_argument("--llama4_dim", type=int, default=5120, 
                       help="Dimension of Llama 4 hidden states")
    parser.add_argument("--csm_dim", type=int, default=4096, 
                       help="Dimension of CSM inputs")
    parser.add_argument("--use_real_llama4", action="store_true", 
                       help="Use real Llama 4 model to generate hidden states")
    parser.add_argument("--fully_synthetic", action="store_true",
                       help="Use fully synthetic data (both inputs and outputs random)")
    parser.add_argument("--noise_level", type=float, default=0.1,
                       help="Noise level for transformation-based data")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for processing")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for model inference")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Generate data
    if args.fully_synthetic:
        # Generate fully synthetic data
        llama4_states, csm_inputs = generate_fully_synthetic_data(
            n_samples=args.n_samples,
            llama4_dim=args.llama4_dim,
            csm_dim=args.csm_dim,
            seed=args.seed
        )
        
        # Store the generation method
        method = "fully_synthetic"
        prompts = ["Synthetic prompt"] * args.n_samples
    else:
        # Generate prompts
        prompts = get_prompt_texts(args.n_samples)
        
        if args.use_real_llama4:
            # Extract real Llama 4 embeddings
            llama4_states = extract_llama4_embeddings(
                model_name=args.llama4_model,
                prompts=prompts,
                device=args.device,
                batch_size=args.batch_size
            )
            method = "real_llama4_synthetic_csm"
        else:
            # Generate synthetic Llama 4 embeddings
            print(f"Generating {args.n_samples} synthetic Llama 4 embeddings...")
            llama4_states = torch.randn(args.n_samples, args.llama4_dim)
            method = "synthetic_transformation"
        
        # Generate synthetic CSM inputs using transformation
        csm_inputs = generate_transformation_based_data(
            llama4_states=llama4_states,
            csm_dim=args.csm_dim,
            noise_level=args.noise_level,
            seed=args.seed
        )
    
    # Verify shapes
    print(f"Llama 4 states shape: {llama4_states.shape}")
    print(f"CSM inputs shape: {csm_inputs.shape}")
    
    # Save data
    print(f"Saving data to {args.output}")
    torch.save({
        "llama4_states": llama4_states,
        "csm_inputs": csm_inputs,
        "prompts": prompts,
        "method": method,
        "args": vars(args)
    }, args.output)
    
    print(f"Data generation complete: {len(llama4_states)} pairs saved to {args.output}")

if __name__ == "__main__":
    main()
