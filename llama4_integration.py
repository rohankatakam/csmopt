#!/usr/bin/env python3
"""
Llama 4 Integration Module

This module provides functionality for integrating Llama 4 with the Sesame CSM
using the trained Llama4Adapter.
"""
import os
import torch
import logging
from typing import Dict, Any, Optional, List, Tuple

from llama4_adapter import Llama4Adapter

logger = logging.getLogger(__name__)

def load_llama4_model(
    model_name: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    device: str = "cuda",
    load_in_4bit: bool = True,
    torch_dtype = None
) -> Tuple[Any, Any]:
    """
    Load Llama 4 model with the correct configuration for CSM integration.
    
    Args:
        model_name: Name of the Llama 4 model to load
        device: Device to load the model on
        load_in_4bit: Whether to use 4-bit quantization (recommended for memory efficiency)
        torch_dtype: Optional torch dtype to use for model loading
        
    Returns:
        Tuple of (model, tokenizer)
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        raise ImportError(
            "Could not import transformers. Please install it with: pip install transformers"
        )
    
    logger.info(f"Loading Llama 4 model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Determine appropriate torch dtype if not specified
    if torch_dtype is None:
        if device == "cuda" and torch.cuda.is_available():
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            torch_dtype = torch.float32
    
    # Set up loading parameters
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "output_hidden_states": True,  # Required to get hidden states for adapter
    }
    
    if device == "cuda" and torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"
        
        # Apply quantization if requested
        if load_in_4bit:
            try:
                model_kwargs["load_in_4bit"] = True
            except Exception as e:
                logger.warning(f"4-bit quantization failed: {e}. Falling back to 8-bit.")
                try:
                    model_kwargs["load_in_8bit"] = True
                    if "load_in_4bit" in model_kwargs:
                        del model_kwargs["load_in_4bit"]
                except Exception as e:
                    logger.warning(f"8-bit quantization failed: {e}. Falling back to 16-bit.")
                    if "load_in_8bit" in model_kwargs:
                        del model_kwargs["load_in_8bit"]
    else:
        # CPU loading
        model_kwargs["device_map"] = None
    
    # Load model
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        logger.info(f"Successfully loaded Llama 4 model with config: {model.config}")
    except Exception as e:
        logger.error(f"Error loading Llama 4 model: {e}")
        raise
    
    return model, tokenizer

def load_llama4_adapter(
    adapter_path: str,
    device: str = "cuda",
    input_dim: int = 5120,
    output_dim: int = 4096
) -> Llama4Adapter:
    """
    Load the Llama 4 adapter from a checkpoint file.
    
    Args:
        adapter_path: Path to the adapter checkpoint file
        device: Device to load the adapter on
        input_dim: Input dimension of the adapter
        output_dim: Output dimension of the adapter
        
    Returns:
        Loaded Llama4Adapter model
    """
    logger.info(f"Loading Llama 4 adapter from: {adapter_path}")
    
    try:
        # First try loading using the class method
        try:
            adapter = Llama4Adapter.load(adapter_path, device=device)
            logger.info("Loaded adapter using Llama4Adapter.load()")
        except (AttributeError, ValueError):
            # Fall back to manual loading
            adapter = Llama4Adapter(input_dim=input_dim, output_dim=output_dim)
            adapter.load_state_dict(torch.load(adapter_path, map_location=device))
            adapter.to(device)
            logger.info("Loaded adapter using manual load_state_dict()")
        
        adapter.eval()  # Set to evaluation mode
        return adapter
    
    except Exception as e:
        logger.error(f"Error loading adapter: {e}")
        raise

class Llama4CSMIntegration:
    """
    Integration class for using Llama 4 with CSM via the adapter.
    """
    def __init__(
        self,
        llama4_model_name: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        adapter_path: str = "checkpoints/llama4_adapter_synthetic_adapter.pt",
        device: str = "cuda",
        load_in_4bit: bool = True
    ):
        """
        Initialize the Llama 4 to CSM integration.
        
        Args:
            llama4_model_name: Name of the Llama 4 model to use
            adapter_path: Path to the adapter checkpoint file
            device: Device to use for computation
            load_in_4bit: Whether to use 4-bit quantization for Llama 4
        """
        self.device = device
        self.llama4_model_name = llama4_model_name
        self.adapter_path = adapter_path
        
        # Load models
        self.model, self.tokenizer = load_llama4_model(
            model_name=llama4_model_name,
            device=device,
            load_in_4bit=load_in_4bit
        )
        
        self.adapter = load_llama4_adapter(
            adapter_path=adapter_path,
            device=device
        )
        
        # Verify dimensions
        expected_input_dim = 5120
        if self.adapter.input_dim != expected_input_dim:
            logger.warning(
                f"Adapter input dimension ({self.adapter.input_dim}) does not match "
                f"expected Llama 4 hidden state dimension ({expected_input_dim})"
            )
    
    def get_adapted_hidden_states(
        self, 
        input_text: str, 
        **generation_kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Get CSM-compatible hidden states from Llama 4 by applying the adapter.
        
        Args:
            input_text: Input text to process
            **generation_kwargs: Additional keyword arguments for text generation
            
        Returns:
            Tuple of (adapted_hidden_states, metadata)
        """
        # Encode input text
        inputs = self.tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Set evaluation mode
        self.model.eval()
        self.adapter.eval()
        
        # Generate with hidden states
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, **generation_kwargs)
            
            # Get the hidden states from the last layer
            last_hidden_states = outputs.hidden_states[-1]
            
            # Apply adapter to get CSM-compatible hidden states
            adapted_hidden_states = self.adapter(last_hidden_states)
        
        # Store metadata for debugging/analysis
        metadata = {
            "input_shape": last_hidden_states.shape,
            "output_shape": adapted_hidden_states.shape,
            "input_text": input_text,
            "adapter_path": self.adapter_path,
            "model_name": self.llama4_model_name
        }
        
        return adapted_hidden_states, metadata

def main():
    """
    Simple demonstration of the Llama 4 integration.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Llama 4 integration with CSM")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-4-Scout-17B-16E-Instruct",
                        help="Llama 4 model name")
    parser.add_argument("--adapter", type=str, default="checkpoints/llama4_adapter_synthetic_adapter.pt",
                        help="Path to adapter checkpoint")
    parser.add_argument("--text", type=str, default="Hello, how are you today?",
                        help="Text to process")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file to save adapted hidden states")
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create integration
    integration = Llama4CSMIntegration(
        llama4_model_name=args.model,
        adapter_path=args.adapter,
        device=args.device
    )
    
    # Get adapted hidden states
    adapted_states, metadata = integration.get_adapted_hidden_states(args.text)
    
    # Print information
    print(f"Input text: {args.text}")
    print(f"Input shape: {metadata['input_shape']}")
    print(f"Output shape: {metadata['output_shape']}")
    
    # Calculate statistics
    mean = adapted_states.mean().item()
    std = adapted_states.std().item()
    min_val = adapted_states.min().item()
    max_val = adapted_states.max().item()
    
    print(f"Statistics of adapted hidden states:")
    print(f"  Mean: {mean:.4f}")
    print(f"  Std: {std:.4f}")
    print(f"  Min: {min_val:.4f}")
    print(f"  Max: {max_val:.4f}")
    
    # Save to file if requested
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        torch.save({
            "hidden_states": adapted_states,
            "metadata": metadata
        }, args.output)
        print(f"Saved adapted hidden states to {args.output}")

if __name__ == "__main__":
    main()
