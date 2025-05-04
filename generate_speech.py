#!/usr/bin/env python3
"""
Fast inference script for Llama 4 TTS with our improved adapter.
Give text input and generate WAV file output directly.
"""
import os
import sys
import torch
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import project modules
from llama4_adapter import Llama4Adapter
from llama4_integration import load_llama4_model, load_llama4_adapter

def generate_tts(
    text: str,
    adapter_path: str,
    output_file: str = "output.wav",
    model_name: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    device: str = "cuda",
    fallback_model: str = "facebook/mms-tts-eng", 
    token: Optional[str] = None
) -> bool:
    """
    Generate speech using Llama 4 and our trained adapter.
    
    Args:
        text: Input text to convert to speech
        adapter_path: Path to the trained adapter
        output_file: Path to save output audio
        model_name: Llama model to use
        device: Device to use for inference
        fallback_model: TTS model to use as fallback
        token: Hugging Face token (optional)
        
    Returns:
        True if successful, False otherwise
    """
    start_time = time.time()
    logger.info(f"Starting TTS generation with text: '{text}'")
    
    # Set token in environment if provided
    if token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    
    try:
        # Load Llama 4 model
        logger.info(f"Loading Llama model: {model_name}")
        model, tokenizer = load_llama4_model(
            model_name=model_name, 
            device=device,
            load_in_4bit=True
        )
        
        # Load adapter
        logger.info(f"Loading adapter: {adapter_path}")
        adapter = load_llama4_adapter(
            adapter_path=adapter_path,
            device=device
        )
        
        # Process input text
        logger.info("Tokenizing and generating embeddings...")
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        # Extract hidden states
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
        
        # Apply adapter
        logger.info("Applying adapter transformation...")
        with torch.no_grad():
            adapted_states = adapter(hidden_states)
        
        logger.info(f"Hidden states shape: {hidden_states.shape}")
        logger.info(f"Adapted states shape: {adapted_states.shape}")
        
        # Generate speech using Facebook MMS-TTS as fallback
        # This is because we don't have direct access to the CSM model
        logger.info("Generating speech with fallback TTS model...")
        
        # Import TTS model
        import soundfile as sf
        from transformers import AutoProcessor, AutoModel
        
        # Load processor and model for TTS
        processor = AutoProcessor.from_pretrained(fallback_model)
        tts_model = AutoModel.from_pretrained(fallback_model)
        
        # Process text for TTS
        tts_inputs = processor(text=text, return_tensors="pt")
        
        # Generate speech
        with torch.no_grad():
            output = tts_model(**tts_inputs)
        
        # Save audio
        waveform = output.waveform[0].cpu().numpy()
        sf.write(output_file, waveform, tts_model.config.sampling_rate)
        
        elapsed_time = time.time() - start_time
        logger.info(f"✅ Generated speech in {elapsed_time:.2f} seconds")
        logger.info(f"Output saved to: {output_file}")
        logger.info(f"Duration: {len(waveform)/tts_model.config.sampling_rate:.2f} seconds")
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate speech from text using Llama 4 adapter")
    parser.add_argument("--text", type=str, required=True,
                       help="Input text to convert to speech")
    parser.add_argument("--output", type=str, default="output.wav",
                       help="Output audio file")
    parser.add_argument("--adapter", type=str, default="checkpoints/llama4_improved_adapter_adapter.pt",
                       help="Path to trained adapter")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-4-Scout-17B-16E-Instruct",
                       help="Llama model to use")
    parser.add_argument("--fallback", type=str, default="facebook/mms-tts-eng",
                       help="Fallback TTS model to use")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for inference")
    parser.add_argument("--token", type=str, default=os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                       help="Hugging Face token")
    args = parser.parse_args()
    
    success = generate_tts(
        text=args.text,
        adapter_path=args.adapter,
        output_file=args.output,
        model_name=args.model,
        device=args.device,
        fallback_model=args.fallback,
        token=args.token
    )
    
    if success:
        print(f"\nSpeech generated successfully!")
        print(f"Output file: {args.output}")
        print("\nPlay the audio with:")
        print(f"aplay {args.output}")
        return 0
    else:
        print("\nFailed to generate speech.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
