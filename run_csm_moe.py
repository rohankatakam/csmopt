"""
Test script for running CSM with optimized vectorized MoE routing.

This script demonstrates how to use the optimized vectorized MoE routing mechanism
with the CSM model for inference on a local .wav file. This implementation
focuses on memory efficiency and performance improvement.
"""

import os
import torch
import torchaudio
import time
import argparse
from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment
from csm_moe_block import create_moe_csm_model
from dataclasses import dataclass

# Disable Triton compilation
os.environ["NO_TORCH_COMPILE"] = "1"

# Default prompts are available at https://hf.co/sesame/csm-1b
prompt_filepath_conversational_a = hf_hub_download(
    repo_id="sesame/csm-1b",
    filename="prompts/conversational_a.wav"
)
prompt_filepath_conversational_b = hf_hub_download(
    repo_id="sesame/csm-1b",
    filename="prompts/conversational_b.wav"
)

SPEAKER_PROMPTS = {
    "conversational_a": {
        "text": (
            "like revising for an exam I'd have to try and like keep up the momentum because I'd "
            "start really early I'd be like okay I'm gonna start revising now and then like "
            "you're revising for ages and then I just like start losing steam I didn't do that "
            "for the exam we had recently to be fair that was a more of a last minute scenario "
            "but like yeah I'm trying to like yeah I noticed this yesterday that like Mondays I "
            "sort of start the day with this not like a panic but like a"
        ),
        "audio": prompt_filepath_conversational_a
    },
    "conversational_b": {
        "text": (
            "like a super Mario level. Like it's very like high detail. And like, once you get "
            "into the park, it just like, everything looks like a computer game and they have all "
            "these, like, you know, if, if there's like a, you know, like in a Mario game, they "
            "will have like a question block. And if you like, you know, punch it, a coin will "
            "come out. So like everyone, when they come into the park, they get like this little "
            "bracelet and then you can go punching question blocks around."
        ),
        "audio": prompt_filepath_conversational_b
    }
}

def load_prompt_audio(audio_path: str, target_sample_rate: int) -> torch.Tensor:
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = audio_tensor.squeeze(0)
    # Resample is lazy so we can always call it
    audio_tensor = torchaudio.functional.resample(
        audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
    )
    return audio_tensor

def prepare_prompt(text: str, speaker: int, audio_path: str, sample_rate: int) -> Segment:
    audio_tensor = load_prompt_audio(audio_path, sample_rate)
    return Segment(text=text, speaker=speaker, audio=audio_tensor)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run CSM with optimized vectorized MoE routing")
    parser.add_argument("--moe_blocks", type=str, default="0,1", 
                        help="Comma-separated list of decoder block indices to apply MoE (default: 0,1)")
    parser.add_argument("--num_experts", type=int, default=8,
                        help="Number of experts in MoE layers (default: 8)")
    parser.add_argument("--top_k", type=int, default=2,
                        help="Number of experts to route to (default: 2)")
    parser.add_argument("--disable_moe", action="store_true",
                        help="Disable MoE routing for comparison")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run in benchmark mode to compare performance")
    parser.add_argument("--memory_profile", action="store_true",
                        help="Track memory usage during generation")
    parser.add_argument("--output", type=str, default="moe_conversation.wav",
                        help="Output wav file name (default: moe_conversation.wav)")
    args = parser.parse_args()

    # Select the best available device, skipping MPS due to float64 limitations
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load model
    print("Loading base CSM model...")
    base_generator = load_csm_1b(device)
    
    # Create optimized vectorized MoE version
    print("Setting up optimized vectorized MoE routing...")
    moe_block_indices = [int(idx) for idx in args.moe_blocks.split(",")]
    print(f"Applying vectorized MoE to decoder blocks: {moe_block_indices}")
    
    moe_model = create_moe_csm_model(
        original_model=base_generator._model,
        moe_block_indices=moe_block_indices,
        num_experts=args.num_experts,
        top_k=args.top_k,
        use_vectorized=True
    )
    
    # Replace the model in the generator
    base_generator._model = moe_model
    
    # Disable MoE if requested
    if args.disable_moe:
        print("Vectorized MoE routing disabled for comparative testing")
        moe_model.enable_moe(False)
    else:
        print(f"Vectorized MoE routing enabled with {args.num_experts} experts, top-{args.top_k} gating")
        
    # Setup memory tracking if requested
    if args.memory_profile and torch.cuda.is_available():
        print("Memory profiling enabled")
        torch.cuda.reset_peak_memory_stats()
        initial_mem = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        print(f"Initial CUDA memory usage: {initial_mem:.2f} MB")

    # Prepare prompts
    prompt_a = prepare_prompt(
        SPEAKER_PROMPTS["conversational_a"]["text"],
        0,
        SPEAKER_PROMPTS["conversational_a"]["audio"],
        base_generator.sample_rate
    )

    prompt_b = prepare_prompt(
        SPEAKER_PROMPTS["conversational_b"]["text"],
        1,
        SPEAKER_PROMPTS["conversational_b"]["audio"],
        base_generator.sample_rate
    )

    # Generate conversation
    conversation = [
        {"text": "Hey how are you doing?", "speaker_id": 0},
        {"text": "Pretty good, pretty good. How about you?", "speaker_id": 1},
        {"text": "I'm great! So happy to be speaking with you today.", "speaker_id": 0},
        {"text": "Me too! This is some cool stuff, isn't it?", "speaker_id": 1}
    ]

    # Generate each utterance
    generated_segments = []
    prompt_segments = [prompt_a, prompt_b]

    if args.benchmark:
        # Run benchmark mode
        print("\nRunning in benchmark mode to compare performance...")
        
        # First run with MoE disabled
        if not args.disable_moe:
            moe_model.enable_moe(False)
            print("\nBenchmarking with MoE disabled:")
            
            start_time = time.time()
            for utterance in conversation:
                print(f"Generating: {utterance['text']}")
                audio_tensor = base_generator.generate(
                    text=utterance['text'],
                    speaker=utterance['speaker_id'],
                    context=prompt_segments + generated_segments,
                    max_audio_length_ms=10_000,
                )
                generated_segments.append(Segment(text=utterance['text'], speaker=utterance['speaker_id'], audio=audio_tensor))
            
            elapsed_no_moe = time.time() - start_time
            print(f"Time without MoE: {elapsed_no_moe:.2f} seconds")
            
            # Reset for the next run
            generated_segments = []
            base_generator._model.reset_caches()
            
            # Now enable vectorized MoE
            moe_model.enable_moe(True)
            print("\nBenchmarking with vectorized MoE enabled:")
        
        # Run with current MoE settings
        start_time = time.time()
        
        if args.memory_profile and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            pre_gen_mem = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        
        for utterance in conversation:
            print(f"Generating: {utterance['text']}")
            audio_tensor = base_generator.generate(
                text=utterance['text'],
                speaker=utterance['speaker_id'],
                context=prompt_segments + generated_segments,
                max_audio_length_ms=10_000,
            )
            generated_segments.append(Segment(text=utterance['text'], speaker=utterance['speaker_id'], audio=audio_tensor))
        
        elapsed_moe = time.time() - start_time
        print(f"Time with vectorized MoE: {elapsed_moe:.2f} seconds")
        
        if args.memory_profile and torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            current_mem = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            print(f"Memory usage - Peak: {peak_mem:.2f} MB, Current: {current_mem:.2f} MB")
            print(f"Memory increase during generation: {current_mem - pre_gen_mem:.2f} MB")
        
        if not args.disable_moe:
            speedup = (elapsed_no_moe / elapsed_moe - 1) * 100
            print(f"Vectorized MoE speedup: {speedup:.2f}%")
    else:
        # Regular generation mode
        for utterance in conversation:
            print(f"Generating: {utterance['text']}")
            audio_tensor = base_generator.generate(
                text=utterance['text'],
                speaker=utterance['speaker_id'],
                context=prompt_segments + generated_segments,
                max_audio_length_ms=10_000,
            )
            generated_segments.append(Segment(text=utterance['text'], speaker=utterance['speaker_id'], audio=audio_tensor))

    # Concatenate all generations
    all_audio = torch.cat([seg.audio for seg in generated_segments], dim=0)
    torchaudio.save(
        args.output,
        all_audio.unsqueeze(0).cpu(),
        base_generator.sample_rate
    )
    print(f"Successfully generated {args.output}")
    
    # Print expert statistics if MoE is enabled
    if not args.disable_moe and hasattr(moe_model, 'get_expert_usage_stats'):
        print("\nVectorized MoE expert usage statistics:")
        stats = moe_model.get_expert_usage_stats()
        for block_name, block_stats in stats.items():
            print(f"\n{block_name}:")
            print(f"  Expert activations: {block_stats.get('expert_activations', [])}")
            print(f"  Expert percentages: {[f'{p:.2f}%' for p in block_stats.get('expert_percentages', [])]}")
            print(f"  Total tokens processed: {block_stats.get('total_tokens_processed', 0)}")
            
        # Show memory usage summary at the end if profiling is enabled
        if args.memory_profile and torch.cuda.is_available():
            print("\nMemory Usage Summary:")
            print(f"Final CUDA memory: {torch.cuda.memory_allocated() / (1024 * 1024):.2f} MB")
            print(f"Peak CUDA memory: {torch.cuda.max_memory_allocated() / (1024 * 1024):.2f} MB")
            torch.cuda.empty_cache()  # Clean up memory at the end

if __name__ == "__main__":
    main()
