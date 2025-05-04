#!/usr/bin/env python3
"""
Generate paired data of Llama 4 hidden states and corresponding CSM inputs.
This data is used to train the Llama 4 to CSM adapter.
"""
import os
import sys
import torch
import torchaudio
import argparse
from tqdm import tqdm
from typing import List, Dict, Tuple
from huggingface_hub import hf_hub_download
from transformers import AutoModelForImageTextToText, AutoProcessor

# Add parent directory to path so we can import from root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generator import load_csm_1b, Segment
from csm_instrumentation import extract_decoder_input_from_csm

def get_prompt_texts(n: int = 5000) -> List[str]:
    """Generate prompt texts for paired data collection"""
    # Simple prompt templates for demonstration
    templates = [
        "Hey, how are you doing?",
        "What's up with you today?",
        "I was wondering if you could help me with something.",
        "Did you hear about the new movie that just came out?",
        "I'm thinking about going on vacation next month.",
        "How was your weekend?",
        "I've been working on a new project lately.",
        "What do you think about the weather today?",
        "I just got back from the store.",
        "Have you tried that new restaurant downtown?",
    ]
    
    # For a real implementation, generate more diverse prompts
    # This is just a placeholder - you'd want many more templates
    # and ways to generate diverse conversation-like prompts
    return templates * (n // len(templates) + 1)[:n]

def load_llama4_model(model_name: str = "meta-llama/Llama-4-Scout-17B-16E-Instruct") -> Tuple[AutoModelForImageTextToText, AutoProcessor]:
    """Load Llama 4 model and tokenizer"""
    print(f"Loading Llama 4 model: {model_name}")
    
    # Load processor (handles both tokenization and image processing)
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Load model with 4-bit quantization for memory efficiency
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True,
        output_hidden_states=True  # Critical: We need the hidden states
    )
    
    return model, processor

def extract_llama4_hidden_states(model, processor, texts: List[str]) -> List[torch.Tensor]:
    """Extract hidden states from Llama 4 for each prompt"""
    hidden_states = []
    batch_size = 4  # Process in small batches to manage memory
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting Llama 4 hidden states"):
        batch_texts = texts[i:i+batch_size]
        # Format as messages for Llama 4
        messages = [[{"role": "user", "content": text}] for text in batch_texts]
        inputs = processor(messages, return_tensors="pt", padding=True).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Get the hidden states from the last layer
            # Shape: [batch_size, seq_len, hidden_size]
            last_hidden_states = outputs.hidden_states[-1]
            
            # Extract hidden state for the last token of each sequence
            for j in range(len(batch_texts)):
                seq_len = inputs.attention_mask[j].sum().item()
                hidden_state = last_hidden_states[j, seq_len - 1].cpu()
                hidden_states.append(hidden_state)
    
    return hidden_states

def extract_csm_inputs(csm_generator, texts: List[str]) -> List[torch.Tensor]:
    """Extract CSM inputs by running Llama 3 + CSM flow"""
    csm_inputs = []
    
    # Use our instrumentation to capture the decoder inputs
    # for each text prompt. This gives us the ground truth target
    # for our adapter training.
    for text in tqdm(texts, desc="Extracting CSM inputs"):
        # Use our instrumentation utility to extract decoder inputs
        csm_input = extract_decoder_input_from_csm(csm_generator, text)
        csm_inputs.append(csm_input)
    
    return csm_inputs

def main():
    parser = argparse.ArgumentParser(description="Generate paired data for Llama 4 to CSM adapter")
    parser.add_argument("--n", type=int, default=5000, help="Number of prompt pairs to generate")
    parser.add_argument("--output", type=str, default="pairs.pt", help="Output file path")
    parser.add_argument("--llama4_model", type=str, default="meta-llama/Llama-4-Scout-17B-16E-Instruct", 
                        help="Llama 4 model name")
    args = parser.parse_args()
    
    # Generate prompt texts
    print(f"Generating {args.n} prompt texts")
    prompts = get_prompt_texts(args.n)
    
    # Load Llama 4 model
    model, tokenizer = load_llama4_model(args.llama4_model)
    
    # Extract Llama 4 hidden states
    llama4_states = extract_llama4_hidden_states(model, tokenizer, prompts)
    
    # Load CSM model
    print("Loading CSM model")
    csm_generator = load_csm_1b()
    
    # Extract CSM inputs
    csm_inputs = extract_csm_inputs(csm_generator, prompts)
    
    # Save paired data
    print(f"Saving {len(llama4_states)} pairs to {args.output}")
    torch.save({
        "llama4_states": torch.stack(llama4_states),
        "csm_inputs": torch.stack(csm_inputs),
        "prompts": prompts
    }, args.output)
    
    print(f"Successfully saved paired data to {args.output}")

if __name__ == "__main__":
    main()
