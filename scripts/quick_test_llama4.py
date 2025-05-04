#!/usr/bin/env python3
"""
Quick test script to verify access to Llama 4 model and extract hidden states.
This is useful for testing access before running the full training pipeline.
"""
import torch
import argparse
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def main():
    parser = argparse.ArgumentParser(description="Quick test for Llama 4 model access")
    parser.add_argument("prompt", type=str, help="Text prompt to test with the model")
    parser.add_argument("--output_states", action="store_true", help="Output hidden states")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B", 
                        help="Model to test (default: Llama-3.2-1B for testing, use meta-llama/Llama-4-Scout-17B-16E-Instruct for actual Llama 4)")
    parser.add_argument("--cpu_offload", action="store_true", help="Enable CPU offloading for large models")
    args = parser.parse_args()
    
    print(f"Testing access to {args.model}...")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Clear CUDA cache first
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Configure quantization based on model size
    print("Loading model...")
    model_kwargs = {
        "output_hidden_states": True,
        "torch_dtype": torch.bfloat16,
    }
    
    # For smaller models like Llama-3.2-1B, we can load directly
    if "Llama-4" in args.model:
        print("Loading large model with quantization...")
        # Configure 4-bit quantization with CPU offloading
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True if args.cpu_offload else False
        )
        model_kwargs.update({
            "quantization_config": quantization_config,
            "device_map": "auto",
            "max_memory": {0: "10GiB", "cpu": "30GiB"}
        })
    else:
        print("Loading smaller model directly...")
    
    # Load the model with appropriate settings
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_kwargs
    )
    
    # Process input (text only in this case)
    print(f"Processing prompt: '{args.prompt}'")
    inputs = tokenizer([args.prompt], return_tensors="pt").to(model.device)
    
    # Generate output
    print("Generating output...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        
    # Process and display results
    output_ids = outputs.sequences[0]
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    print("\n--- Model Output ---")
    print(output_text)
    
    # Check hidden states if requested
    if args.output_states:
        # Get hidden states from the first output token
        # In a sequence generation task, hidden states are returned differently
        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            # Access structure depends on the exact model and configuration
            try:
                # Format varies between models, this is for typical format
                hidden_states = outputs.hidden_states
                
                print("\n--- Hidden States Information ---")
                print(f"Number of generation steps with hidden states: {len(hidden_states)}")
                
                # Check the first generation step
                first_step = hidden_states[0]
                if isinstance(first_step, tuple):
                    # Some models return hidden states as tuples of tensors
                    print(f"Number of layers: {len(first_step)}")
                    last_layer_states = first_step[-1]
                    print(f"Last layer shape: {last_layer_states.shape}")
                    print(f"Hidden dimension: {last_layer_states.shape[-1]}")
                    
                    # Check if it's 8192 as expected for Llama 4
                    if last_layer_states.shape[-1] == 8192:
                        print("\n✅ SUCCESS: Hidden dimension is 8192 as expected for Llama 4!")
                    else:
                        print(f"\n⚠️ WARNING: Hidden dimension is {last_layer_states.shape[-1]}, not 8192 as expected!")
                else:
                    print("Hidden states structure is not as expected. Please investigate.")
            except Exception as e:
                print(f"Error processing hidden states: {e}")
                print("Hidden states structure: ", type(outputs.hidden_states))
        else:
            print("\n⚠️ No hidden states found in the output!")
    
    print("\nTest completed!")

if __name__ == "__main__":
    main()
