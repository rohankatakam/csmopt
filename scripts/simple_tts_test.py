#!/usr/bin/env python3
"""
Simple TTS test using a reliable TTS model from Hugging Face.
This script will generate audio without requiring an adapter.
"""
import os
import sys
import torch
import time
import logging
import argparse
import soundfile as sf
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def generate_speech(
    text: str,
    output_file: str = "tts_output.wav",
    model_name: str = "microsoft/speecht5_tts",
    token: str = None,
    device: str = "cuda"
):
    """Generate speech using a reliable TTS model."""
    
    # Set token in environment if provided
    if token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    
    try:
        logger.info(f"Loading model {model_name}...")
        
        # Import required libraries
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
        import numpy as np
        
        # Load processor and model
        processor = SpeechT5Processor.from_pretrained(model_name)
        model = SpeechT5ForTextToSpeech.from_pretrained(model_name).to(device)
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)
        
        # Load speaker embeddings
        embeddings_dataset = torch.load("https://huggingface.co/datasets/Matthijs/cmu-arctic-xvectors/resolve/main/xvector_tsv.pt")
        speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(device)
        
        # Process text input
        inputs = processor(text=text, return_tensors="pt").to(device)
        
        # Generate speech with voice cloning
        speech = model.generate_speech(
            inputs["input_ids"], 
            speaker_embeddings, 
            vocoder=vocoder
        )
        
        # Save output
        sf.write(output_file, speech.cpu().numpy(), 16000)
        
        logger.info(f"Audio generated successfully: {output_file}")
        logger.info(f"Duration: {len(speech)/16000:.2f} seconds")
        
        return True
        
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        
        # Try alternative model if first one fails
        try:
            logger.info("Trying alternative model...")
            
            # Import required libraries
            from transformers import VitsModel, AutoTokenizer
            
            # Load model and tokenizer
            model = VitsModel.from_pretrained("facebook/mms-tts-eng").to(device)
            tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
            
            # Process input
            inputs = tokenizer(text=text, return_tensors="pt").to(device)
            
            # Generate speech
            with torch.no_grad():
                output = model(**inputs)
            
            # Get waveform from output
            waveform = output.waveform[0].cpu().numpy()
            
            # Save audio
            sf.write(output_file, waveform, model.config.sampling_rate)
            
            logger.info(f"Audio generated successfully with alternative model: {output_file}")
            logger.info(f"Duration: {len(waveform)/model.config.sampling_rate:.2f} seconds")
            
            return True
            
        except Exception as e2:
            logger.error(f"Alternative model also failed: {e2}")
            
            # Final fallback - use TorToiSe TTS if available
            try:
                logger.info("Trying TorToiSe TTS as final fallback...")
                
                # Use TorToiSe TTS
                from tortoise.api import TextToSpeech
                from tortoise.utils.audio import load_audio, load_voice
                import torchaudio
                
                # Initialize model
                tts = TextToSpeech()
                
                # Generate speech
                gen = tts.tts(text, voice_samples=None, conditioning_latents=None)
                
                # Save audio
                torchaudio.save(output_file, gen.squeeze(0).cpu(), 24000)
                
                logger.info(f"Audio generated successfully with TorToiSe: {output_file}")
                
                return True
                
            except Exception as e3:
                logger.error(f"All TTS attempts failed: {e3}")
                return False

def main():
    parser = argparse.ArgumentParser(description="Simple TTS Test")
    parser.add_argument("--text", type=str, default="Hello! This is a test of text to speech synthesis. I hope you can hear this clearly.",
                      help="Text to convert to speech")
    parser.add_argument("--output", type=str, default="simple_tts_output.wav",
                      help="Output audio file")
    parser.add_argument("--token", type=str, default=os.environ.get("HUGGING_FACE_HUB_TOKEN"),
                      help="Hugging Face token")
    parser.add_argument("--model", type=str, default="microsoft/speecht5_tts",
                      help="TTS model to use")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to use for inference")
    args = parser.parse_args()
    
    # Generate speech
    success = generate_speech(
        text=args.text,
        output_file=args.output,
        model_name=args.model,
        token=args.token,
        device=args.device
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
