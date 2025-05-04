#!/usr/bin/env python3
"""
Run CSM with Llama 4 Adapter

This script provides command-line integration for running the CSM model
with Llama 4 as the backbone, using the trained adapter to transform
Llama 4's hidden states to CSM's expected format.
"""
import os
import sys
import torch
import argparse
import logging
import time
from typing import Dict, Any, Optional

# Add parent directory to path to import from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from llama4_integration import Llama4CSMIntegration
from llama4_adapter import Llama4Adapter
from logging_config import setup_logger

# Configure logging
logger = setup_logger('run_csm_llama4', level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Run CSM with Llama 4 adapter")
    
    # Model options
    parser.add_argument("--llama4_model", type=str, 
                       default="meta-llama/Llama-4-Scout-17B-16E-Instruct",
                       help="Llama 4 model name")
    parser.add_argument("--adapter", type=str, 
                       default="checkpoints/llama4_adapter_synthetic_adapter.pt",
                       help="Path to adapter checkpoint")
    parser.add_argument("--csm_model", type=str, 
                       default="sesame-csm-1b",
                       help="CSM model name")
    
    # Input/output options
    parser.add_argument("--input", type=str, required=True,
                       help="Input text or file (use @filename to read from file)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output audio file (default: output.wav)")
    parser.add_argument("--output_hidden_states", type=str, default=None,
                       help="Output file for hidden states (optional)")
    
    # Performance options
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda, cpu)")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                       help="Use 4-bit quantization for Llama 4")
    parser.add_argument("--kv_cache", action="store_true", default=True,
                       help="Enable KV cache for faster generation")
    
    # Debug options
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run in benchmark mode to measure performance")
    parser.add_argument("--performance_stats", action="store_true",
                       help="Log performance statistics")
    
    args = parser.parse_args()
    
    # Set debug mode if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
    
    # Read input from file if specified with @
    if args.input.startswith("@"):
        input_file = args.input[1:]
        logger.info(f"Reading input from file: {input_file}")
        with open(input_file, "r") as f:
            input_text = f.read().strip()
    else:
        input_text = args.input
    
    # Set default output file
    if args.output is None:
        args.output = "output.wav"
    
    try:
        # Initialize Llama 4 integration
        logger.info("Initializing Llama 4 integration")
        integration = Llama4CSMIntegration(
            llama4_model_name=args.llama4_model,
            adapter_path=args.adapter,
            device=args.device,
            load_in_4bit=args.load_in_4bit
        )
        
        # Generate hidden states with adapter
        logger.info("Generating adapted hidden states from Llama 4")
        start_time = time.time()
        end_time = time.time()
        
        # Start timer
        if args.performance_stats and torch.cuda.is_available():
            start_time = time.time()
        
        # Get adapted hidden states
        adapted_states, metadata = integration.get_adapted_hidden_states(
            input_text,
            use_cache=args.kv_cache
        )
        
        # End timer for hidden state generation
        if args.performance_stats and torch.cuda.is_available():
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f"Hidden state generation and adaptation took {elapsed_time:.2f} seconds")
        
        # Save hidden states if requested
        if args.output_hidden_states:
            os.makedirs(os.path.dirname(os.path.abspath(args.output_hidden_states)), exist_ok=True)
            torch.save({
                "hidden_states": adapted_states,
                "metadata": metadata
            }, args.output_hidden_states)
            logger.info(f"Saved hidden states to {args.output_hidden_states}")
        
        # Load CSM model and generate audio
        logger.info(f"Loading CSM model: {args.csm_model}")
        # In a real implementation, we would now:
        # 1. Load the CSM decoder
        # 2. Pass the adapted hidden states to it
        # 3. Generate audio
        # 
        # This is a placeholder for that functionality
        logger.info("Placeholder: Passing adapted hidden states to CSM model for audio generation")
        logger.info(f"Would save audio output to {args.output}")
        
        # In benchmark mode, run multiple iterations to measure performance
        if args.benchmark:
            logger.info("Running benchmark")
            n_iterations = 10
            
            if torch.cuda.is_available():
                start_time = time.time()
                end_time = time.time()
                
                # Warmup
                for _ in range(3):
                    integration.get_adapted_hidden_states(input_text, use_cache=args.kv_cache)
                
                # Benchmark
                start_time = time.time()
                for _ in range(n_iterations):
                    integration.get_adapted_hidden_states(input_text, use_cache=args.kv_cache)
                end_time = time.time()
                
                elapsed_time = end_time - start_time
                avg_time = elapsed_time / n_iterations
                
                logger.info(f"Benchmark results:")
                logger.info(f"  Average processing time: {avg_time:.2f} seconds per iteration")
                logger.info(f"  Total time for {n_iterations} iterations: {elapsed_time:.2f} seconds")
        
        logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error running CSM with Llama 4: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
