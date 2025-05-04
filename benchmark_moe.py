#!/usr/bin/env python3
"""
Benchmark script for CSM MoE Phase 2 optimizations.

This script runs benchmarks to measure the performance improvements from:
1. Mixed precision (FP16/BF16)
2. Optimized routing algorithms
3. Different expert configurations

Results are saved as CSV files and visualizations.
"""

import os
import torch
import torchaudio
import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from generator import load_csm_1b, Segment
from csm_moe_block import create_moe_csm_model
from huggingface_hub import hf_hub_download

# Ensure output directory exists
os.makedirs("benchmark_results", exist_ok=True)

# Default prompts
prompt_filepath_a = hf_hub_download(
    repo_id="sesame/csm-1b",
    filename="prompts/conversational_a.wav"
)

def load_prompt_audio(audio_path: str, target_sample_rate: int) -> torch.Tensor:
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = audio_tensor.squeeze(0)
    # Resample if needed
    audio_tensor = torchaudio.functional.resample(
        audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
    )
    return audio_tensor

def prepare_prompt(text: str, speaker: int, audio_path: str, sample_rate: int) -> Segment:
    audio_tensor = load_prompt_audio(audio_path, sample_rate)
    return Segment(text=text, speaker=speaker, audio=audio_tensor)

@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    name: str
    num_experts: int
    top_k: int
    precision: str
    routing_algorithm: str
    moe_blocks: List[int] = field(default_factory=lambda: [0, 1])
    memory_profile: bool = True
    
    def get_cmd_args(self) -> List[str]:
        """Convert config to command-line arguments."""
        args = [
            f"--num_experts={self.num_experts}",
            f"--top_k={self.top_k}",
            f"--precision={self.precision}",
            f"--routing_algorithm={self.routing_algorithm}",
            f"--moe_blocks={','.join(map(str, self.moe_blocks))}"
        ]
        if self.memory_profile:
            args.append("--memory_profile")
        
        return args
    
    def __str__(self) -> str:
        return (f"{self.name}: {self.num_experts} experts, top-{self.top_k}, "
                f"{self.precision}, {self.routing_algorithm} routing")

@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    config: BenchmarkConfig
    generation_time: float
    peak_memory_mb: Optional[float] = None
    expert_usage_stats: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        memory_str = f", Peak Memory: {self.peak_memory_mb:.2f} MB" if self.peak_memory_mb else ""
        return f"{self.config}, Time: {self.generation_time:.2f}s{memory_str}"

def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run a single benchmark with the given configuration."""
    print(f"\n=== Running benchmark: {config} ===")
    
    # Create model and generator
    base_generator = load_csm_1b()
    base_model = base_generator._model
    
    # Track time for model creation
    start_time = time.time()
    
    # Create MoE model with the specified configuration
    moe_model = create_moe_csm_model(
        original_model=base_model,
        moe_block_indices=config.moe_blocks,
        num_experts=config.num_experts,
        top_k=config.top_k,
        precision=config.precision,
        routing_algorithm=config.routing_algorithm
    )
    
    # Replace model in generator
    base_generator._model = moe_model
    
    # Reset any CUDA memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    # Prepare prompt
    prompt_segment = prepare_prompt(
        text="like revising for an exam I'd have to try and like keep up the momentum",
        speaker=0,
        audio_path=prompt_filepath_a,
        sample_rate=base_generator.sample_rate
    )
    
    # Test utterances
    test_utterances = [
        {"text": "Hi, how are you doing today?", "speaker_id": 0},
        {"text": "I'm doing well, thanks for asking. How about you?", "speaker_id": 1},
    ]
    
    # Track generation time
    generation_start = time.time()
    
    # Generate utterances
    context_segments = [prompt_segment]
    for utterance in test_utterances:
        audio_tensor = base_generator.generate(
            text=utterance["text"],
            speaker=utterance["speaker_id"],
            context=context_segments,
            max_audio_length_ms=5000,
        )
        context_segments.append(Segment(
            text=utterance["text"], 
            speaker=utterance["speaker_id"], 
            audio=audio_tensor
        ))
    
    generation_time = time.time() - generation_start
    
    # Get expert usage statistics
    expert_stats = {}
    if hasattr(moe_model, 'get_expert_usage_stats'):
        expert_stats = moe_model.get_expert_usage_stats()
    
    # Get peak memory usage
    peak_memory = None
    if torch.cuda.is_available() and config.memory_profile:
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
    
    # Clean up
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return BenchmarkResult(
        config=config,
        generation_time=generation_time,
        peak_memory_mb=peak_memory,
        expert_usage_stats=expert_stats
    )

def save_results_to_csv(results: List[BenchmarkResult], filename: str) -> None:
    """Save benchmark results to a CSV file."""
    data = []
    for result in results:
        row = {
            "name": result.config.name,
            "num_experts": result.config.num_experts,
            "top_k": result.config.top_k,
            "precision": result.config.precision,
            "routing_algorithm": result.config.routing_algorithm,
            "generation_time": result.generation_time,
            "peak_memory_mb": result.peak_memory_mb or 0
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def plot_benchmark_results(results_df: pd.DataFrame, output_dir: str) -> None:
    """
    Plot benchmark results.
    
    Args:
        results_df: DataFrame with benchmark results
        output_dir: Directory to save plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Skip plotting if we don't have enough results
    if len(results_df) == 0:
        print("No benchmark results to plot.")
        return
    
    # Extract data for successful benchmarks only (where we have timing data)
    valid_results = results_df.dropna(subset=['generation_time'])
    
    if len(valid_results) == 0:
        print("No valid benchmark results to plot.")
        return
        
    # Plot time vs name
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    
    # Extract data
    names = valid_results['name'].tolist()
    times = valid_results['generation_time'].tolist()
    memory = valid_results['peak_memory_mb'].tolist()
    
    # Set x positions
    index = np.arange(len(names))
    
    # Create bars
    plt.bar(index, times, bar_width, label='Time (s)')
    plt.bar(index + bar_width, memory, bar_width, label='Memory (MB)')
    
    # Add labels and title
    plt.xlabel('Configuration')
    plt.ylabel('Value')
    plt.title('MoE Benchmark Results: Time vs Memory')
    plt.xticks(index + bar_width / 2, names, rotation=45, ha='right')
    plt.legend()
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_results.png'))
    plt.close()
    
    # Plot expert counts for each benchmark (if available)
    for name in names:
        if f"{name}_expert_counts" in valid_results.columns:
            expert_counts = valid_results.loc[valid_results['name'] == name, f"{name}_expert_counts"].iloc[0]
            if expert_counts and isinstance(expert_counts, dict):
                plt.figure(figsize=(10, 5))
                expert_ids = list(range(len(expert_counts)))
                counts = [expert_counts.get(f"expert_{i}", 0) for i in expert_ids]
                plt.bar(expert_ids, counts)
                plt.title(f'Expert Usage for {name}')
                plt.xlabel('Expert ID')
                plt.ylabel('Token Count')
                plt.savefig(os.path.join(output_dir, f'expert_usage_{name}.png'))
                plt.close()

def main():
    # Define benchmark configurations
    benchmark_configs = [
        # Baseline (FP32)
        BenchmarkConfig(
            name="baseline",
            num_experts=8,
            top_k=2,
            precision="fp32",
            routing_algorithm="top_k"
        ),
        
        # Mixed Precision Tests
        BenchmarkConfig(
            name="mixed_precision_fp16",
            num_experts=8,
            top_k=2,
            precision="fp16",
            routing_algorithm="top_k"
        ),
        
        # Only run bf16 if CUDA is available and on a compatible device
        BenchmarkConfig(
            name="mixed_precision_bf16",
            num_experts=8,
            top_k=2,
            precision="bf16" if torch.cuda.is_available() and hasattr(torch, 'bfloat16') else "fp16",
            routing_algorithm="top_k"
        ),
        
        # Routing Algorithm Tests
        BenchmarkConfig(
            name="balanced_routing",
            num_experts=8,
            top_k=2,
            precision="fp16",
            routing_algorithm="balanced"
        ),
        
        BenchmarkConfig(
            name="expert_choice_routing",
            num_experts=8,
            top_k=2,
            precision="fp16",
            routing_algorithm="expert_choice"
        ),
        
        # Expert Count Tests
        BenchmarkConfig(
            name="4_experts",
            num_experts=4,
            top_k=2,
            precision="fp16",
            routing_algorithm="balanced"
        ),
        
        BenchmarkConfig(
            name="16_experts",
            num_experts=16,
            top_k=2,
            precision="fp16",
            routing_algorithm="balanced"
        )
    ]
    
    # Run benchmarks
    results = []
    for config in benchmark_configs:
        try:
            result = run_benchmark(config)
            results.append(result)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error running benchmark {config}: {e}")
    
    # Save results
    save_results_to_csv(results, "benchmark_results/moe_benchmarks.csv")
    
    # Create DataFrame for plotting
    if results:
        results_df = pd.DataFrame([
            {
                'name': r.config.name,
                'generation_time': r.generation_time,
                'peak_memory_mb': r.peak_memory_mb or 0,
                f"{r.config.name}_expert_counts": r.expert_usage_stats.get('expert_activations', {})
            } for r in results
        ])
        
        # Plot results
        plot_benchmark_results(results_df, "benchmark_results")
    
    print("Benchmarking complete!")

if __name__ == "__main__":
    main()
