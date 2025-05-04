#!/usr/bin/env python3
"""
Test CSM TTS functionality with the available model.
This script specifically tests the Sesame CSM model for TTS generation.
"""
import os
import sys
import torch
import logging
import soundfile as sf
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def attempt_csm_tts(
    text: str,
    output_file: str = "output_csm_tts.wav",
    model_name: str = "sesame/csm-1b",
    device: str = "cuda",
    adapter_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Attempt to generate speech using the CSM model.
    
    Args:
        text: Text to convert to speech
        output_file: Path to save the output audio
        model_name: Name of the CSM model
        device: Device to use for inference
        adapter_path: Optional path to a Llama adapter
        
    Returns:
        Dictionary with results
    """
    try:
        logger.info(f"Attempting TTS with {model_name}...")
        
        # Try different approaches to load and use the model
        # Approach 1: Use transformers pipeline
        try:
            from transformers import pipeline
            logger.info("Trying transformers pipeline approach...")
            
            # Try to use the TTS pipeline
            tts = pipeline("text-to-speech", model=model_name, device=device)
            
            # Generate speech
            speech = tts(text)
            
            # Save output
            if "sampling_rate" in speech:
                sf.write(output_file, speech["audio"], speech["sampling_rate"])
                sampling_rate = speech["sampling_rate"]
            else:
                sf.write(output_file, speech["audio"], 24000)  # CSM likely uses 24kHz
                sampling_rate = 24000
                
            return {
                "status": "success",
                "method": "pipeline",
                "output_file": output_file,
                "sampling_rate": sampling_rate,
                "duration": len(speech["audio"]) / sampling_rate
            }
        except Exception as e:
            logger.info(f"Pipeline approach failed: {e}")
            logger.info("Trying direct model approach...")
            
            # Approach 2: Load model and processor directly
            from transformers import AutoProcessor, AutoModelForTextToWaveform, AutoModelForTextToSpeech
            
            # Try different model types since we're not certain of the exact class
            model_types = [
                (AutoModelForTextToSpeech, "AutoModelForTextToSpeech"),
                (AutoModelForTextToWaveform, "AutoModelForTextToWaveform"),
                # Add more potential model types as needed
            ]
            
            # Try all model types
            for model_cls, model_type_name in model_types:
                try:
                    logger.info(f"Trying {model_type_name}...")
                    
                    processor = AutoProcessor.from_pretrained(model_name)
                    model = model_cls.from_pretrained(model_name).to(device)
                    
                    # Process the input text
                    inputs = processor(text=text, return_tensors="pt").to(device)
                    
                    # Generate speech
                    with torch.no_grad():
                        output = model.generate(**inputs)
                    
                    # Process output
                    if hasattr(output, "waveform"):
                        # If output is an object with waveform attribute
                        audio = output.waveform.cpu().numpy().squeeze()
                    else:
                        # If output is a tensor
                        audio = output.cpu().numpy().squeeze()
                    
                    # Save audio
                    sampling_rate = getattr(model.config, "sampling_rate", 24000)
                    sf.write(output_file, audio, sampling_rate)
                    
                    return {
                        "status": "success",
                        "method": model_type_name,
                        "output_file": output_file,
                        "sampling_rate": sampling_rate,
                        "duration": len(audio) / sampling_rate
                    }
                except Exception as e:
                    logger.info(f"{model_type_name} approach failed: {e}")
            
            # Approach 3: Try to use a lower-level implementation
            logger.info("Trying to work with raw model configuration...")
            
            # Import the potential model
            from transformers import AutoConfig, AutoModel, AutoTokenizer
            
            # Get model config
            config = AutoConfig.from_pretrained(model_name)
            logger.info(f"Model config type: {config.model_type}")
            
            # We need to adapt to the specific model type that CSM uses
            # This requires some reverse engineering from the config
            
            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(device)
            
            # Tokenize input
            inputs = tokenizer(text, return_tensors="pt").to(device)
            
            # This is where we'd need to design a custom implementation
            # based on the actual CSM model
            logger.warning("Direct implementation for this specific model not available.")
            logger.warning("Need more detailed documentation of the CSM API.")
            
            # Fallback - look for any useful functions in the model
            for method_name in dir(model):
                if "generate" in method_name.lower() or "speech" in method_name.lower():
                    logger.info(f"Found potentially useful method: {method_name}")
            
        # No success with known approaches
        return {
            "status": "failed",
            "error": "All approaches failed. Need specific implementation for this model."
        }
        
    except Exception as e:
        logger.error(f"Failed to generate TTS: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }

def test_with_fallback_tts():
    """Test with a fallback TTS model that is more likely to work."""
    try:
        import torch
        from tortoise.api import TextToSpeech
        from tortoise.utils.audio import load_audio, load_voice, generate_voice_seconds
        import torchaudio
        
        logger.info("Testing with Tortoise TTS (local fallback)...")
        
        # Initialize TTS model
        tts = TextToSpeech()
        
        # Generate speech
        text = "Hello, this is a test of the Tortoise text to speech system. I hope you can hear this clearly."
        
        # Generate speech with preset voice
        voice_samples = None  # Default voice
        conditioning_latents = None  # Default settings
        
        gen = tts.tts(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents)
        
        # Save output
        output_file = "fallback_tts_output.wav"
        torchaudio.save(output_file, gen.squeeze(0).cpu(), 24000)
        
        logger.info(f"✅ Successfully generated fallback TTS at {output_file}")
        return {
            "status": "success",
            "output_file": output_file
        }
    except Exception as e:
        logger.error(f"Fallback TTS failed: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }

def try_facebook_mms_tts():
    """Try Facebook's MMS-TTS model which is generally accessible."""
    try:
        logger.info("Testing Facebook MMS-TTS model...")
        
        # Import required libraries
        from transformers import AutoProcessor, AutoModel
        
        # Model name
        model_name = "facebook/mms-tts-eng"
        
        # Load processor and model
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Process input
        text = "Hello, this is a test of the Facebook MMS text to speech system. I hope you can hear this clearly."
        inputs = processor(text=text, return_tensors="pt")
        
        # Generate speech
        output = model.generate(**inputs)
        
        # Save audio
        output_file = "mms_tts_output.wav"
        sf.write(output_file, output[0].numpy(), model.config.sampling_rate)
        
        logger.info(f"✅ Successfully generated MMS-TTS audio at {output_file}")
        return {
            "status": "success",
            "output_file": output_file
        }
    except Exception as e:
        logger.error(f"Facebook MMS-TTS failed: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Test CSM TTS functionality")
    parser.add_argument("--text", type=str, default="Hello, this is a test of the CSM text to speech system. I hope you can hear this clearly.",
                       help="Text to convert to speech")
    parser.add_argument("--output", type=str, default="csm_tts_output.wav",
                       help="Output audio file path")
    parser.add_argument("--model", type=str, default="sesame/csm-1b",
                       help="CSM model name")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda, cpu)")
    parser.add_argument("--adapter", type=str, default=None,
                       help="Optional path to a Llama adapter")
    parser.add_argument("--try_all", action="store_true",
                       help="Try all TTS options, including fallbacks")
    args = parser.parse_args()
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Try CSM TTS
    csm_result = attempt_csm_tts(
        text=args.text,
        output_file=args.output,
        model_name=args.model,
        device=args.device,
        adapter_path=args.adapter
    )
    
    # Print result
    if csm_result["status"] == "success":
        logger.info(f"✅ CSM TTS generation successful!")
        logger.info(f"  - Method: {csm_result['method']}")
        logger.info(f"  - Output: {csm_result['output_file']}")
        logger.info(f"  - Duration: {csm_result['duration']:.2f}s")
        logger.info(f"  - Sampling rate: {csm_result['sampling_rate']} Hz")
        
        # Print command to play audio
        print("\nPlay the generated audio with:")
        print(f"aplay {csm_result['output_file']}")
        
        return 0
    
    logger.warning("CSM TTS generation failed, trying fallbacks...")
    
    # Try Facebook MMS-TTS
    mms_result = try_facebook_mms_tts()
    
    if mms_result["status"] == "success":
        logger.info("✅ Facebook MMS-TTS generation successful!")
        logger.info(f"  - Output: {mms_result['output_file']}")
        
        # Print command to play audio
        print("\nPlay the generated audio with:")
        print(f"aplay {mms_result['output_file']}")
        
        return 0
    
    # Try with Tortoise fallback (if installed)
    if args.try_all:
        fallback_result = test_with_fallback_tts()
        
        if fallback_result["status"] == "success":
            logger.info("✅ Fallback TTS generation successful!")
            logger.info(f"  - Output: {fallback_result['output_file']}")
            
            # Print command to play audio
            print("\nPlay the generated audio with:")
            print(f"aplay {fallback_result['output_file']}")
            
            return 0
    
    logger.error("❌ All TTS attempts failed.")
    return 1

if __name__ == "__main__":
    sys.exit(main())
