"""
Memory profiling utilities for tracking GPU memory usage during model execution.
This helps analyze the memory efficiency of different model configurations.
"""

import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable

class MemoryProfiler:
    """
    Tracks GPU memory usage during model execution.
    """
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.timestamps = []
        self.allocated_memory = []
        self.reserved_memory = []
        self.events = {}
        self.start_time = None
        self.is_profiling = False
    
    def start(self):
        """Start profiling memory usage"""
        if not torch.cuda.is_available() and self.device == "cuda":
            print("CUDA not available, memory profiling disabled")
            return
        
        self.timestamps = []
        self.allocated_memory = []
        self.reserved_memory = []
        self.events = {}
        self.start_time = time.time()
        self.is_profiling = True
        
        # Reset peak stats at the start
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            # Initial reading
            self._record_memory("start")
    
    def stop(self):
        """Stop profiling memory usage"""
        self.is_profiling = False
        if self.device == "cuda" and torch.cuda.is_available():
            # Final reading
            self._record_memory("end")
    
    def record(self, event_name: str):
        """
        Record memory usage at a specific point
        
        Args:
            event_name: Name of the event for labeling
        """
        if not self.is_profiling:
            return
        
        if self.device == "cuda" and torch.cuda.is_available():
            self._record_memory(event_name)
    
    def _record_memory(self, event_name: str):
        """Internal function to record memory stats"""
        if not torch.cuda.is_available():
            return
            
        current_time = time.time() - self.start_time
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)    # MB
        
        self.timestamps.append(current_time)
        self.allocated_memory.append(allocated)
        self.reserved_memory.append(reserved)
        self.events[event_name] = len(self.timestamps) - 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get collected memory statistics
        
        Returns:
            Dictionary of memory statistics
        """
        if self.device != "cuda" or not torch.cuda.is_available():
            return {
                "device": self.device,
                "peak_allocated": 0,
                "timestamps": [],
                "allocated": [],
                "reserved": [],
                "events": {}
            }
            
        peak_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
        
        return {
            "device": self.device,
            "peak_allocated": peak_allocated,
            "timestamps": self.timestamps,
            "allocated": self.allocated_memory,
            "reserved": self.reserved_memory,
            "events": self.events
        }
    
    def print_summary(self):
        """Print a summary of memory usage"""
        if self.device != "cuda" or not torch.cuda.is_available():
            print("Memory profiling not available (CUDA not available)")
            return
            
        stats = self.get_stats()
        
        print("\nMemory Usage Summary:")
        print(f"Peak allocated memory: {stats['peak_allocated']:.2f} MB")
        
        if len(self.timestamps) > 0:
            print(f"Initial allocated memory: {self.allocated_memory[0]:.2f} MB")
            print(f"Final allocated memory: {self.allocated_memory[-1]:.2f} MB")
            print(f"Change in allocated memory: {self.allocated_memory[-1] - self.allocated_memory[0]:.2f} MB")
            
            if len(self.events) > 0:
                print("\nMemory at key events:")
                for event, idx in self.events.items():
                    print(f"  {event}: {self.allocated_memory[idx]:.2f} MB allocated, "
                          f"{self.reserved_memory[idx]:.2f} MB reserved")
    
    def plot(self, title: str = "Memory Usage", filename: Optional[str] = None):
        """
        Plot memory usage over time
        
        Args:
            title: Title for the plot
            filename: If provided, save plot to this file
            
        Returns:
            The matplotlib figure object
        """
        if self.device != "cuda" or not torch.cuda.is_available() or len(self.timestamps) == 0:
            print("No memory data available to plot")
            return None
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(self.timestamps, self.allocated_memory, 'b-', label='Allocated Memory (MB)')
        ax.plot(self.timestamps, self.reserved_memory, 'r--', label='Reserved Memory (MB)')
        
        # Add event markers
        for event, idx in self.events.items():
            ax.axvline(x=self.timestamps[idx], color='g', linestyle='-', alpha=0.5)
            ax.text(self.timestamps[idx], max(self.allocated_memory) * 1.05, event,
                   rotation=90, verticalalignment='bottom')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Memory (MB)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        if filename:
            plt.savefig(filename, dpi=200, bbox_inches='tight')
            print(f"Memory profile plot saved to {filename}")
        
        return fig

def profile_function(
    func: Callable, 
    *args,
    device: str = "cuda",
    title: str = "Memory Profile",
    filename: Optional[str] = None,
    **kwargs
) -> Tuple[Any, Dict[str, Any]]:
    """
    Profile memory usage during execution of a function
    
    Args:
        func: Function to profile
        *args: Arguments to pass to the function
        device: Device to profile
        title: Title for the plot
        filename: If provided, save plot to this file
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Tuple of (function result, memory statistics)
    """
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, running without memory profiling")
        return func(*args, **kwargs), {}
    
    profiler = MemoryProfiler(device)
    profiler.start()
    
    # Record pre-execution
    profiler.record("pre_execution")
    
    # Execute function
    result = func(*args, **kwargs)
    
    # Record post-execution
    profiler.record("post_execution")
    profiler.stop()
    
    # Print summary
    profiler.print_summary()
    
    # Plot if filename provided
    if filename:
        profiler.plot(title, filename)
    
    return result, profiler.get_stats()
