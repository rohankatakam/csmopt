#!/usr/bin/env python3
"""
Simple test script for Llama 4 TTS integration.
This script runs a basic test of the Llama 4 to CSM integration.
"""
import os
import torch
import argparse
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import project modules
from llama4_integration import Llama4CSMIntegration
from llama4_adapter import Llama4Adapter

def main():
    parser = argparse.ArgumentParser(description="Test Llama 4 to CSM TTS integration")
    
    # Model options
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B",
                      help="Model to use (default: meta-llama/Llama-3.2-1B - smaller and faster)")
    parser.add_argument("--adapter", type=str, 
                      default="checkpoints/llama4_adapter_synthetic_adapter.pt",
                      help="Path to adapter checkpoint")
    parser.add_argument("--text", type=str, 
                      default="Hello, this is a test of the Llama 4 to CSM integration.",
                      help="Text to synthesize")
    parser.add_argument("--device", type=str, default="cuda",
                      help="Device to use (cuda, cpu)")
    
    args = parser.parse_args()
    
    logger.info(f"Using model: {args.model}")
    logger.info(f"Using adapter: {args.adapter}")
    logger.info(f"Text to synthesize: '{args.text}'")
    
    try:
        # Initialize the integration
        logger.info("Initializing Llama 4 to CSM integration...")
        integration = Llama4CSMIntegration(
            llama4_model_name=args.model,
            adapter_path=args.adapter,
            device=args.device,
            load_in_4bit=True
        )
        
        # Get adapted hidden states
        logger.info("Getting adapted hidden states...")
        hidden_states, metadata = integration.get_adapted_hidden_states(args.text)
        
        # Log information about the hidden states
        logger.info(f"Hidden states shape: {hidden_states.shape}")
        logger.info(f"Hidden states dtype: {hidden_states.dtype}")
        logger.info(f"Hidden states range: [{hidden_states.min().item():.4f}, {hidden_states.max().item():.4f}]")
        logger.info(f"Hidden states mean: {hidden_states.mean().item():.4f}")
        logger.info(f"Hidden states std: {hidden_states.std().item():.4f}")
        
        # At this point, we would pass the hidden states to the CSM model
        # Since we don't have access to the CSM model's decoder, we'll just log the success
        logger.info("Successfully generated adapted hidden states!")
        logger.info("In a complete setup, these would be passed to the CSM decoder for audio generation.")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
