"""
Utility functions for instrumenting the CSM model to extract internal states.
These functions are used for the Llama 4 integration to generate training data 
for the adapter that maps between Llama 4 and CSM hidden states.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from contextlib import contextmanager

from models import Model

class DecoderInputCapture:
    """
    Module to capture and store inputs to the CSM decoder.
    This is used to collect training data for the Llama 4 adapter.
    """
    def __init__(self):
        self.captured_inputs = []
        self.is_capturing = False
        self.hooks = []
    
    def start_capture(self):
        """Start capturing decoder inputs"""
        self.captured_inputs = []
        self.is_capturing = True
    
    def stop_capture(self):
        """Stop capturing decoder inputs"""
        self.is_capturing = False
    
    def get_captured_inputs(self) -> List[torch.Tensor]:
        """Get list of captured decoder inputs"""
        return self.captured_inputs
    
    def capture_hook(self, module, input, output):
        """Hook function to capture decoder inputs"""
        if self.is_capturing:
            # First element of input tuple contains the actual input tensor
            if isinstance(input, tuple) and len(input) > 0:
                # Clone and detach to avoid any gradient or memory issues
                self.captured_inputs.append(input[0].detach().cpu())

    def register_hooks(self, model: Model):
        """Register hooks on the decoder to capture inputs"""
        self.remove_hooks()  # Clean up any existing hooks
        
        # Attach hooks to the decoder layers
        if hasattr(model.decoder, 'layers') and len(model.decoder.layers) > 0:
            for i, layer in enumerate(model.decoder.layers):
                # For CSM model based on LLaMA, we want to capture inputs to each decoder layer
                hook = layer.register_forward_hook(self.capture_hook)
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

@contextmanager
def capture_decoder_inputs(model: Model) -> DecoderInputCapture:
    """
    Context manager to capture decoder inputs from a CSM model.
    
    Usage:
    ```
    with capture_decoder_inputs(model) as capture:
        output = model.generate(...)
        
    decoder_inputs = capture.get_captured_inputs()
    ```
    
    Args:
        model: CSM model to instrument
        
    Returns:
        DecoderInputCapture: Object containing captured decoder inputs
    """
    capture = DecoderInputCapture()
    capture.register_hooks(model)
    capture.start_capture()
    try:
        yield capture
    finally:
        capture.stop_capture()
        capture.remove_hooks()

def extract_decoder_input_from_csm(generator, text: str) -> torch.Tensor:
    """
    Extract decoder input from CSM generator for a given text.
    This is used to generate training data for the Llama 4 adapter.
    
    Args:
        generator: CSM generator
        text: Input text
        
    Returns:
        Tensor with shape (4096,) representing input to the CSM decoder
    """
    model = generator._model
    
    # Capture decoder inputs
    with capture_decoder_inputs(model) as capture:
        # Generate with a simple text prompt
        # We don't care about the output, just want to capture the decoder inputs
        with torch.no_grad():
            generator.generate(
                text=text,
                speaker=0,  # Default speaker
                context=[],
                max_audio_length_ms=1000,  # Keep short for efficiency
            )
        
    # Process captured inputs
    captured_inputs = capture.get_captured_inputs()
    
    if not captured_inputs:
        print("Warning: No decoder inputs captured. Using zeros.")
        return torch.zeros(4096, dtype=torch.float32)
    
    # Take the first input to the first decoder layer
    # This represents the output of the LLaMA backbone and input to the decoder
    decoder_input = captured_inputs[0][0]  # First batch, first token
    
    # Return as a flat tensor of the expected dimensionality
    return decoder_input
