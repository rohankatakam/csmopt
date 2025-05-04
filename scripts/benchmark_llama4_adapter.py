#!/usr/bin/env python3
"""
Benchmark Llama 4 Adapter Integration

This script performs comprehensive benchmarking of the Llama 4 to CSM
integration pipeline, measuring memory usage, latency, and throughput
at different stages of the process.
"""
import os
import sys
import time
import torch
import numpy as np
import argparse
import logging
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

# Add parent directory to path to import from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from llama4_integration import Llama4CSMIntegration, load_llama4_model, load_llama4_adapter
from llama4_adapter import Llama4Adapter

# Assuming logging_config.py is in the parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from logging_config import setup_logger

# Setup logger
logger = setup_logger('llama4_adapter_benchmark', level=logging.INFO)

class PerformanceMonitor:
    """Monitor various performance metrics during the benchmark."""
    
    def __init__(self, enable_cuda: bool = True):
        """
        Initialize the performance monitor.
        
        Args:
            enable_cuda: Whether to track CUDA metrics (if available)
        """
        self.enable_cuda = enable_cuda and torch.cuda.is_available()
        self.process = psutil.Process()
        self.metrics = {
            "timestamps": [],
            "cpu_percent": [],
            "memory_usage_mb": [],
            "cuda_memory_allocated_mb": [],
            "cuda_memory_reserved_mb": []
        }
        
        # Track starting memory usage to compute deltas
        self.initial_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        if self.enable_cuda:
            torch.cuda.reset_peak_memory_stats()
            self.initial_cuda_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            self.initial_cuda_reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
    
    def start_monitoring(self, interval: float = 0.1):
        """Start background monitoring of performance metrics."""
        # Reset metrics
        for key in self.metrics:
            self.metrics[key] = []
        
        # Store initial state
        self.record_metrics()
    
    def record_metrics(self):
        """Record current performance metrics."""
        self.metrics["timestamps"].append(time.time())
        self.metrics["cpu_percent"].append(self.process.cpu_percent())
        self.metrics["memory_usage_mb"].append(self.process.memory_info().rss / (1024 * 1024))
        
        if self.enable_cuda:
            self.metrics["cuda_memory_allocated_mb"].append(
                torch.cuda.memory_allocated() / (1024 * 1024)
            )
            self.metrics["cuda_memory_reserved_mb"].append(
                torch.cuda.memory_reserved() / (1024 * 1024)
            )
        else:
            self.metrics["cuda_memory_allocated_mb"].append(0)
            self.metrics["cuda_memory_reserved_mb"].append(0)
    
    def get_peak_memory(self) -> Dict[str, float]:
        """Get peak memory usage during monitoring."""
        if not self.metrics["memory_usage_mb"]:
            return {
                "peak_cpu_memory_mb": 0,
                "peak_cuda_allocated_mb": 0,
                "peak_cuda_reserved_mb": 0
            }
        
        return {
            "peak_cpu_memory_mb": max(self.metrics["memory_usage_mb"]),
            "peak_cuda_allocated_mb": max(self.metrics["cuda_memory_allocated_mb"]) if self.enable_cuda else 0,
            "peak_cuda_reserved_mb": max(self.metrics["cuda_memory_reserved_mb"]) if self.enable_cuda else 0
        }
    
    def get_memory_delta(self) -> Dict[str, float]:
        """Get memory usage delta from the beginning of monitoring."""
        if not self.metrics["memory_usage_mb"]:
            return {
                "delta_cpu_memory_mb": 0,
                "delta_cuda_allocated_mb": 0,
                "delta_cuda_reserved_mb": 0
            }
        
        last_memory = self.metrics["memory_usage_mb"][-1]
        
        result = {
            "delta_cpu_memory_mb": last_memory - self.initial_memory,
        }
        
        if self.enable_cuda:
            last_cuda_allocated = self.metrics["cuda_memory_allocated_mb"][-1]
            last_cuda_reserved = self.metrics["cuda_memory_reserved_mb"][-1]
            
            result.update({
                "delta_cuda_allocated_mb": last_cuda_allocated - self.initial_cuda_allocated,
                "delta_cuda_reserved_mb": last_cuda_reserved - self.initial_cuda_reserved
            })
        else:
            result.update({
                "delta_cuda_allocated_mb": 0,
                "delta_cuda_reserved_mb": 0
            })
        
        return result
    
    def get_average_cpu_percent(self) -> float:
        """Get average CPU usage during monitoring."""
        if not self.metrics["cpu_percent"]:
            return 0.0
        return sum(self.metrics["cpu_percent"]) / len(self.metrics["cpu_percent"])
    
    def plot_metrics(self, output_path: str):
        """
        Plot performance metrics over time.
        
        Args:
            output_path: Path to save the plot
        """
        if not self.metrics["timestamps"]:
            logger.warning("No metrics to plot")
            return
        
        # Convert timestamps to relative seconds
        start_time = self.metrics["timestamps"][0]
        relative_times = [t - start_time for t in self.metrics["timestamps"]]
        
        # Create figure with multiple subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot CPU percent
        axs[0].plot(relative_times, self.metrics["cpu_percent"], 'b-')
        axs[0].set_title('CPU Usage')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('CPU (%)')
        axs[0].grid(True)
        
        # Plot memory usage
        axs[1].plot(relative_times, self.metrics["memory_usage_mb"], 'r-')
        axs[1].set_title('CPU Memory Usage')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Memory (MB)')
        axs[1].grid(True)
        
        # Plot CUDA memory
        axs[2].plot(relative_times, self.metrics["cuda_memory_allocated_mb"], 'g-', 
                   label='Allocated')
        axs[2].plot(relative_times, self.metrics["cuda_memory_reserved_mb"], 'y-',
                   label='Reserved')
        axs[2].set_title('CUDA Memory Usage')
        axs[2].set_xlabel('Time (s)')
        axs[2].set_ylabel('Memory (MB)')
        axs[2].grid(True)
        axs[2].legend()
        
        # Adjust layout and save
        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Performance metrics plot saved to {output_path}")

def measure_latency(
    func, 
    args: list = None, 
    kwargs: dict = None, 
    warmup: int = 3, 
    repeats: int = 10,
    use_cuda: bool = True
) -> Dict[str, float]:
    """
    Measure the latency of a function with warmup runs.
    
    Args:
        func: Function to measure
        args: Positional arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        warmup: Number of warmup runs
        repeats: Number of measurement runs
        use_cuda: Whether to use CUDA events for timing (more accurate for GPU)
        
    Returns:
        Dictionary with latency statistics
    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    
    # Warmup runs
    for _ in range(warmup):
        func(*args, **kwargs)
    
    # Use CUDA events for more accurate GPU timing
    if use_cuda and torch.cuda.is_available():
        times = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            func(*args, **kwargs)
            end.record()
            
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))  # milliseconds
        
        times_ms = times
    else:
        # Fallback to Python timing
        times = []
        for _ in range(repeats):
            start_time = time.time()
            func(*args, **kwargs)
            end_time = time.time()
            times.append(end_time - start_time)
        
        times_ms = [t * 1000 for t in times]  # convert to milliseconds
    
    return {
        "mean_ms": np.mean(times_ms),
        "median_ms": np.median(times_ms),
        "std_ms": np.std(times_ms),
        "min_ms": np.min(times_ms),
        "max_ms": np.max(times_ms),
        "times_ms": times_ms
    }

def benchmark_integration(
    integration: Llama4CSMIntegration,
    input_texts: List[str],
    output_dir: str,
    enable_cuda_metrics: bool = True
) -> Dict[str, Any]:
    """
    Benchmark the Llama 4 to CSM integration pipeline.
    
    Args:
        integration: Llama4CSMIntegration instance
        input_texts: List of input texts to process
        output_dir: Directory to save benchmark results
        enable_cuda_metrics: Whether to track CUDA metrics
        
    Returns:
        Dictionary with benchmark results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Summary dictionary for results
    results = {
        "latency_ms": {},
        "throughput": {},
        "memory": {},
        "model_info": {
            "llama4_model": integration.llama4_model_name,
            "adapter_path": integration.adapter_path,
            "adapter_params": sum(p.numel() for p in integration.adapter.parameters()),
            "device": integration.device
        }
    }
    
    # 1. Measure adapter latency
    logger.info("Measuring adapter forward pass latency...")
    random_input = torch.randn(1, 5120).to(integration.device)
    
    adapter_latency = measure_latency(
        integration.adapter,
        args=[random_input],
        warmup=5,
        repeats=100,
        use_cuda=enable_cuda_metrics
    )
    results["latency_ms"]["adapter_forward"] = adapter_latency
    
    # 2. Measure end-to-end processing latency
    logger.info("Measuring end-to-end processing latency...")
    
    # Monitor memory during processing
    monitor = PerformanceMonitor(enable_cuda=enable_cuda_metrics)
    
    all_latencies = []
    token_counts = []
    
    for i, text in enumerate(input_texts):
        logger.info(f"Processing input {i+1}/{len(input_texts)}: {text[:50]}...")
        
        # Tokenize to count tokens
        n_tokens = len(integration.tokenizer.encode(text))
        token_counts.append(n_tokens)
        
        # Measure latency
        monitor.start_monitoring()
        
        latency = measure_latency(
            integration.get_adapted_hidden_states,
            args=[text],
            warmup=1,
            repeats=3,
            use_cuda=enable_cuda_metrics
        )
        
        monitor.record_metrics()  # Record final state
        
        # Store latency
        all_latencies.append(latency)
        
        # Record memory usage
        peak_memory = monitor.get_peak_memory()
        memory_delta = monitor.get_memory_delta()
        
        # Plot the metrics for this input
        monitor.plot_metrics(os.path.join(output_dir, f"metrics_input_{i+1}.png"))
        
        # Compute throughput
        throughput = n_tokens / (latency["mean_ms"] / 1000)  # tokens per second
        
        # Store individual results
        results.setdefault("per_input", []).append({
            "input_text": text,
            "n_tokens": n_tokens,
            "latency_ms": latency["mean_ms"],
            "throughput_tokens_per_sec": throughput,
            "peak_memory": peak_memory,
            "memory_delta": memory_delta
        })
    
    # Compute aggregate statistics
    results["latency_ms"]["end_to_end"] = {
        "mean_ms": np.mean([l["mean_ms"] for l in all_latencies]),
        "median_ms": np.median([l["mean_ms"] for l in all_latencies]),
        "std_ms": np.std([l["mean_ms"] for l in all_latencies]),
        "min_ms": np.min([l["mean_ms"] for l in all_latencies]),
        "max_ms": np.max([l["mean_ms"] for l in all_latencies])
    }
    
    # Compute throughput statistics
    token_per_sec_values = [
        entry["throughput_tokens_per_sec"] for entry in results["per_input"]
    ]
    results["throughput"]["tokens_per_second"] = {
        "mean": np.mean(token_per_sec_values),
        "median": np.median(token_per_sec_values),
        "min": np.min(token_per_sec_values),
        "max": np.max(token_per_sec_values)
    }
    
    # Get per-token latency
    token_latencies = [
        entry["latency_ms"] / entry["n_tokens"] for entry in results["per_input"]
    ]
    results["latency_ms"]["per_token"] = {
        "mean_ms": np.mean(token_latencies),
        "median_ms": np.median(token_latencies),
        "min_ms": np.min(token_latencies),
        "max_ms": np.max(token_latencies)
    }
    
    # Extract memory statistics
    peak_memories = [entry["peak_memory"] for entry in results["per_input"]]
    results["memory"]["peak"] = {
        "cpu_memory_mb": max(entry["peak_cpu_memory_mb"] for entry in peak_memories),
        "cuda_allocated_mb": max(entry["peak_cuda_allocated_mb"] for entry in peak_memories),
        "cuda_reserved_mb": max(entry["peak_cuda_reserved_mb"] for entry in peak_memories)
    }
    
    # Save results to file
    results_file = os.path.join(output_dir, "benchmark_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate summary report
    report = generate_benchmark_report(results, output_dir)
    
    return results

def generate_benchmark_report(results: Dict[str, Any], output_dir: str) -> str:
    """
    Generate a human-readable benchmark report.
    
    Args:
        results: Dictionary with benchmark results
        output_dir: Directory to save the report
        
    Returns:
        Path to the generated report
    """
    report_path = os.path.join(output_dir, "benchmark_report.md")
    
    with open(report_path, "w") as f:
        f.write("# Llama 4 to CSM Integration Benchmark Report\n\n")
        
        # Model info
        f.write("## Model Information\n\n")
        f.write(f"- **Llama 4 Model**: {results['model_info']['llama4_model']}\n")
        f.write(f"- **Adapter Parameters**: {results['model_info']['adapter_params']:,}\n")
        f.write(f"- **Device**: {results['model_info']['device']}\n\n")
        
        # Latency
        f.write("## Latency Measurements\n\n")
        
        f.write("### Adapter Forward Pass\n\n")
        f.write(f"- **Mean**: {results['latency_ms']['adapter_forward']['mean_ms']:.2f} ms\n")
        f.write(f"- **Median**: {results['latency_ms']['adapter_forward']['median_ms']:.2f} ms\n")
        f.write(f"- **Min**: {results['latency_ms']['adapter_forward']['min_ms']:.2f} ms\n")
        f.write(f"- **Max**: {results['latency_ms']['adapter_forward']['max_ms']:.2f} ms\n\n")
        
        f.write("### End-to-End Processing\n\n")
        f.write(f"- **Mean**: {results['latency_ms']['end_to_end']['mean_ms']:.2f} ms\n")
        f.write(f"- **Median**: {results['latency_ms']['end_to_end']['median_ms']:.2f} ms\n")
        f.write(f"- **Min**: {results['latency_ms']['end_to_end']['min_ms']:.2f} ms\n")
        f.write(f"- **Max**: {results['latency_ms']['end_to_end']['max_ms']:.2f} ms\n\n")
        
        f.write("### Per-Token Latency\n\n")
        f.write(f"- **Mean**: {results['latency_ms']['per_token']['mean_ms']:.2f} ms\n")
        f.write(f"- **Median**: {results['latency_ms']['per_token']['median_ms']:.2f} ms\n\n")
        
        # Throughput
        f.write("## Throughput\n\n")
        f.write(f"- **Mean**: {results['throughput']['tokens_per_second']['mean']:.2f} tokens/second\n")
        f.write(f"- **Median**: {results['throughput']['tokens_per_second']['median']:.2f} tokens/second\n")
        f.write(f"- **Range**: {results['throughput']['tokens_per_second']['min']:.2f} - ")
        f.write(f"{results['throughput']['tokens_per_second']['max']:.2f} tokens/second\n\n")
        
        # Memory usage
        f.write("## Memory Usage\n\n")
        f.write(f"- **Peak CPU Memory**: {results['memory']['peak']['cpu_memory_mb']:.2f} MB\n")
        f.write(f"- **Peak CUDA Allocated**: {results['memory']['peak']['cuda_allocated_mb']:.2f} MB\n")
        f.write(f"- **Peak CUDA Reserved**: {results['memory']['peak']['cuda_reserved_mb']:.2f} MB\n\n")
        
        # Per-input details
        f.write("## Per-Input Performance\n\n")
        f.write("| Input | Tokens | Latency (ms) | Throughput (tokens/s) |\n")
        f.write("|-------|--------|-------------|----------------------|\n")
        
        for entry in results["per_input"]:
            text = entry["input_text"][:30].replace("\n", " ")
            if len(entry["input_text"]) > 30:
                text += "..."
            
            f.write(f"| \"{text}\" | {entry['n_tokens']} | {entry['latency_ms']:.2f} | ")
            f.write(f"{entry['throughput_tokens_per_sec']:.2f} |\n")
    
    logger.info(f"Benchmark report saved to {report_path}")
    return report_path

def main():
    parser = argparse.ArgumentParser(description="Benchmark Llama 4 to CSM adapter integration")
    
    # Model options
    parser.add_argument("--llama4_model", type=str, 
                       default="meta-llama/Llama-4-Scout-17B-16E-Instruct",
                       help="Llama 4 model name")
    parser.add_argument("--adapter", type=str, 
                       default="checkpoints/llama4_adapter_synthetic_adapter.pt",
                       help="Path to adapter checkpoint")
    
    # Benchmark options
    parser.add_argument("--inputs", type=str, default=None,
                       help="Path to file with input texts (one per line)")
    parser.add_argument("--output_dir", type=str, default="benchmarks",
                       help="Directory to save benchmark results")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda, cpu)")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                       help="Use 4-bit quantization for Llama 4")
    
    args = parser.parse_args()
    
    # Default test inputs if not provided
    if args.inputs is None:
        test_inputs = [
            "Hello, how are you today?",
            "What's the weather like in San Francisco?",
            "Can you tell me about the history of artificial intelligence?",
            "I'm thinking about learning to play the guitar. Any advice on getting started?",
            "Here's a longer request that talks about multiple topics. First, I'm interested in learning more about quantum computing. Second, I'd like to know how machine learning algorithms work. Finally, can you recommend some good books on philosophy?"
        ]
    else:
        # Read inputs from file
        with open(args.inputs, "r") as f:
            test_inputs = [line.strip() for line in f if line.strip()]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize integration
    logger.info("Initializing Llama 4 integration...")
    integration = Llama4CSMIntegration(
        llama4_model_name=args.llama4_model,
        adapter_path=args.adapter,
        device=args.device,
        load_in_4bit=args.load_in_4bit
    )
    
    # Run benchmarks
    logger.info("Running benchmarks...")
    results = benchmark_integration(
        integration=integration,
        input_texts=test_inputs,
        output_dir=args.output_dir
    )
    
    logger.info(f"Benchmarking complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
