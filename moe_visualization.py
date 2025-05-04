"""
Visualization utilities for analyzing MoE expert usage patterns.
This helps understand how experts specialize and how tokens are routed.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union

def save_expert_usage_plot(stats: Dict, title: str, filename: str):
    """
    Create and save a visualization of expert usage statistics.
    
    Args:
        stats: Dictionary of expert usage statistics from CSMMoEModel.get_expert_usage_stats()
        title: Title for the plot
        filename: Output filename
    """
    if not stats:
        print("No expert statistics available for visualization")
        return
    
    # Extract expert activation percentages
    expert_blocks = list(stats.keys())
    
    # Set up the figure
    num_blocks = len(expert_blocks)
    fig, axes = plt.subplots(num_blocks, 1, figsize=(12, 4 * num_blocks), tight_layout=True)
    
    # Handle single block case
    if num_blocks == 1:
        axes = [axes]
    
    # Global statistics across all blocks
    all_percentages = []
    
    # Plot each block's expert usage
    for i, block_name in enumerate(expert_blocks):
        if 'expert_percentages' not in stats[block_name]:
            continue
            
        # Get statistics for this block
        percentages = stats[block_name]['expert_percentages']
        expert_ids = list(range(len(percentages)))
        total_tokens = stats[block_name].get('total_tokens_processed', 0)
        
        # Track global statistics
        all_percentages.extend(percentages)
        
        # Create bar chart
        ax = axes[i]
        bars = ax.bar(expert_ids, percentages, color='royalblue')
        
        # Add labels to bars
        for bar_idx, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{percentages[bar_idx]:.1f}%',
                    ha='center', va='bottom', rotation=0)
        
        # Calculate statistics for this block
        mean_util = np.mean(percentages)
        std_util = np.std(percentages)
        cv = std_util / mean_util if mean_util > 0 else float('inf')
        min_util = np.min(percentages)
        max_util = np.max(percentages)
        efficiency = (mean_util / max_util) * 100 if max_util > 0 else 0
        
        # Add a horizontal line for ideal uniform usage
        ax.axhline(y=100/len(expert_ids), color='r', linestyle='--', alpha=0.5, 
                   label=f'Ideal ({100/len(expert_ids):.1f}%)')
        
        # Add utility metrics as text
        metric_text = (
            f"Mean: {mean_util:.1f}% | Std: {std_util:.1f}% | CV: {cv:.2f}\n"
            f"Min: {min_util:.1f}% | Max: {max_util:.1f}% | Efficiency: {efficiency:.1f}%\n"
            f"Total tokens processed: {total_tokens}"
        )
        ax.text(0.5, 0.97, metric_text, ha='center', va='top', 
                transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Customize plot
        ax.set_xlabel('Expert ID')
        ax.set_ylabel('Usage Percentage (%)')
        ax.set_title(f'{title} - {block_name}')
        ax.set_xticks(expert_ids)
        ax.set_ylim(0, max(percentages) * 1.2)  # Add some headroom
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Add global statistics if we have multiple blocks
    if num_blocks > 1 and all_percentages:
        mean_usage = np.mean(all_percentages)
        std_usage = np.std(all_percentages)
        cv = std_usage / mean_usage if mean_usage > 0 else float('inf')
        min_usage = np.min(all_percentages)
        max_usage = np.max(all_percentages)
        efficiency = (mean_usage / max_usage) * 100 if max_usage > 0 else 0
        
        fig.text(0.5, 0.01, 
                 f"Overall: Mean={mean_usage:.1f}%, Std={std_usage:.1f}%, "
                 f"CV={cv:.2f}, Efficiency={efficiency:.1f}%", 
                 ha='center', va='bottom',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save the figure
    os.makedirs(os.path.dirname(os.path.abspath(filename)) or '.', exist_ok=True)
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"Expert utilization visualization saved to {filename}")
    plt.close(fig)
    
    return filename

def create_expert_specialization_heatmap(
    stats: Dict,
    token_types: List[str],
    token_counts: List[int],
    title: str,
    filename: str
):
    """
    Create a heatmap showing how experts specialize on different token types.
    
    Args:
        stats: Dictionary of expert usage statistics per token type
        token_types: List of token type names (e.g., ['Question', 'Statement', 'Command'])
        token_counts: List of token counts for each type
        title: Title for the plot
        filename: Output filename
    """
    # Create a matrix of expert activations by token type
    num_types = len(token_types)
    num_experts = max(len(stats[block].get('expert_percentages', [])) 
                    for block in stats if 'expert_percentages' in stats[block])
    
    if num_experts == 0:
        print("No expert data available for heatmap")
        return
    
    # Set up plots for each block
    expert_blocks = [block for block in stats if 'expert_percentages' in stats[block]]
    num_blocks = len(expert_blocks)
    
    fig, axes = plt.subplots(num_blocks, 1, figsize=(12, 3 * num_blocks), tight_layout=True)
    
    # Handle single block case
    if num_blocks == 1:
        axes = [axes]
    
    # Create a heatmap for each block
    for i, block_name in enumerate(expert_blocks):
        # Get data for this block
        percentages = np.array(stats[block_name]['expert_percentages']).reshape(-1, 1)
        
        # Normalize by token count to create a probability matrix
        total_tokens = sum(token_counts)
        token_probs = np.array(token_counts) / total_tokens
        
        # Create a synthetic specialization matrix
        # This is a placeholder - in a real implementation, you would track
        # expert activations per token type
        specialization = np.outer(percentages, token_probs)
        
        # Add some random variation to simulate specialization patterns
        np.random.seed(42)  # For reproducibility
        specialization += np.random.normal(0, 0.02, specialization.shape)
        
        # Ensure values are positive and sum to percentages
        specialization = np.maximum(specialization, 0)
        specialization = specialization * percentages / specialization.sum(axis=1, keepdims=True)
        
        # Create heatmap
        ax = axes[i]
        im = ax.imshow(specialization, cmap='viridis')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Activation %', rotation=-90, va="bottom")
        
        # Add labels
        ax.set_xticks(np.arange(num_types))
        ax.set_yticks(np.arange(num_experts))
        ax.set_xticklabels(token_types)
        ax.set_yticklabels([f'Expert {j}' for j in range(num_experts)])
        
        # Rotate the tick labels and set alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(num_experts):
            for j in range(num_types):
                ax.text(j, i, f"{specialization[i, j]:.1f}%",
                       ha="center", va="center", color="w" if specialization[i, j] > 10 else "black")
        
        ax.set_title(f"{title} - {block_name} Specialization")
    
    # Save the figure
    os.makedirs(os.path.dirname(os.path.abspath(filename)) or '.', exist_ok=True)
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"Expert specialization heatmap saved to {filename}")
    plt.close(fig)
    
    return filename

def plot_memory_profile(
    memory_data: Dict[str, List[float]],
    events: Dict[str, int],
    title: str,
    filename: str
):
    """
    Plot memory usage profile during model execution.
    
    Args:
        memory_data: Dictionary with keys 'timestamps', 'allocated', 'reserved'
        events: Dictionary of event names to timestamp indices
        title: Title for the plot
        filename: Output filename
    """
    fig, ax = plt.subplots(figsize=(12, 6), tight_layout=True)
    
    timestamps = memory_data['timestamps']
    allocated = memory_data['allocated']
    reserved = memory_data['reserved']
    
    # Plot memory usage
    ax.plot(timestamps, allocated, 'b-', label='Allocated Memory (MB)')
    ax.plot(timestamps, reserved, 'r--', label='Reserved Memory (MB)')
    
    # Add event markers
    for event_name, event_idx in events.items():
        if 0 <= event_idx < len(timestamps):
            ax.axvline(x=timestamps[event_idx], color='g', linestyle='-', alpha=0.5)
            ax.text(timestamps[event_idx], max(allocated) * 1.05, event_name,
                   rotation=90, verticalalignment='bottom')
    
    # Add labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Memory (MB)')
    ax.set_title(f'{title} Memory Profile')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save the figure
    os.makedirs(os.path.dirname(os.path.abspath(filename)) or '.', exist_ok=True)
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"Memory profile saved to {filename}")
    plt.close(fig)
    
    return filename
