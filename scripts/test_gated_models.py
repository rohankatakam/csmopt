#!/usr/bin/env python3
"""
Test access to gated Hugging Face models using the provided token.
This script verifies access to Llama 4, Llama 3.2, and CSM models.
"""
import os
import sys
import torch
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def test_model_access(model_name: str, token: str = None):
    """
    Test access to a specific model on Hugging Face.
    
    Args:
        model_name: The name of the model to test
        token: Optional Hugging Face token
        
    Returns:
        Dictionary with test results
    """
    logger.info(f"Testing access to {model_name}...")
    start_time = time.time()
    
    try:
        # Import the required libraries
        from transformers import AutoConfig, AutoTokenizer
        
        # Set token in environment if provided
        if token:
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        
        # Try to load the model config (lightweight operation)
        config = AutoConfig.from_pretrained(model_name, token=token)
        
        # Try to load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        
        # Calculate access time
        access_time = time.time() - start_time
        
        # Get model details
        model_type = getattr(config, "model_type", "unknown")
        hidden_size = getattr(config, "hidden_size", -1)
        vocab_size = getattr(config, "vocab_size", -1)
        
        logger.info(f"✅ Successfully accessed {model_name} in {access_time:.2f}s")
        logger.info(f"  - Model type: {model_type}")
        logger.info(f"  - Hidden size: {hidden_size}")
        logger.info(f"  - Vocabulary size: {vocab_size}")
        
        # Test tokenization
        if hasattr(tokenizer, "encode"):
            sample_text = "Hello, I'm testing access to this model."
            tokens = tokenizer.encode(sample_text)
            logger.info(f"  - Sample tokenization length: {len(tokens)}")
            
        return {
            "status": "success",
            "access_time": access_time,
            "model_type": model_type,
            "hidden_size": hidden_size,
            "vocab_size": vocab_size,
        }
        
    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"❌ Failed to access {model_name}: {str(e)}")
        return {
            "status": "failed",
            "access_time": error_time,
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Test access to gated Hugging Face models")
    parser.add_argument("--token", type=str, default=os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                       help="Hugging Face token (default: from environment)")
    args = parser.parse_args()
    
    # Models to test
    models = [
        "meta-llama/Llama-4-Scout-17B-16E-Instruct",  # Llama 4
        "meta-llama/Llama-3.2-1B",                   # Llama 3.2
        "sesame/csm-1b"                              # CSM model
    ]
    
    # Test each model
    results = {}
    for model in models:
        results[model] = test_model_access(model, args.token)
    
    # Print summary
    print("\n=== Access Test Results ===")
    for model, result in results.items():
        status_icon = "✅" if result["status"] == "success" else "❌"
        print(f"{status_icon} {model}: {result['status']} ({result['access_time']:.2f}s)")
        
        if result["status"] == "success":
            print(f"  - Model type: {result['model_type']}")
            print(f"  - Hidden size: {result['hidden_size']}")
            print(f"  - Vocabulary size: {result['vocab_size']}")
        else:
            print(f"  - Error: {result['error']}")
    
    # Check overall access
    access_count = sum(1 for r in results.values() if r["status"] == "success")
    if access_count == len(models):
        logger.info("✅ Successfully accessed all models!")
        return 0
    elif access_count > 0:
        logger.warning(f"⚠️ Partial access: {access_count}/{len(models)} models accessible.")
        return 0
    else:
        logger.error("❌ Failed to access any models.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
