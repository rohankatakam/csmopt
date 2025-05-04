import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class Llama4Adapter(nn.Module):
    """
    Adapter to transform Llama 4's 5120-d hidden states to 
    match Sesame CSM's expected 4096-d input format.
    """
    def __init__(self, 
                 input_dim: int = 5120, 
                 output_dim: int = 4096, 
                 hidden_dim: Optional[int] = None,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim or output_dim
        self.dtype = dtype
        
        # Two-layer MLP with GELU activation
        self.down_proj = nn.Linear(input_dim, output_dim, dtype=dtype)
        self.activation = nn.GELU()
        self.out_proj = nn.Linear(output_dim, output_dim, dtype=dtype)
        
        # Initialize weights for better convergence
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform Llama 4 hidden states to Sesame CSM compatible format.
        
        Args:
            x: Input tensor of shape [..., input_dim]
            
        Returns:
            Transformed tensor of shape [..., output_dim]
        """
        # Ensure adapter weights match input dtype
        if x.dtype != self.down_proj.weight.dtype:
            logger.info(f"Converting adapter weights from {self.down_proj.weight.dtype} to {x.dtype}")
            self.to(x.dtype)

        x = self.down_proj(x)
        x = self.activation(x)
        x = self.out_proj(x)
        return x
    
    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "Llama4Adapter":
        """
        Load adapter from checkpoint file.
        
        Args:
            path: Path to checkpoint file
            device: Device to load model to
            
        Returns:
            Loaded adapter model
        """
        state_dict = torch.load(path, map_location=device)
        
        # Handle various checkpoint formats
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        
        # Get model configuration from state dict
        if "config" in state_dict:
            config = state_dict["config"]
            model = cls(**config)
            model.load_state_dict(state_dict["model"])
        else:
            # Try to infer dimensions from keys
            keys = list(state_dict.keys())
            if "down_proj.weight" in keys:
                input_dim = state_dict["down_proj.weight"].shape[1]
                output_dim = state_dict["down_proj.weight"].shape[0]
                model = cls(input_dim=input_dim, output_dim=output_dim)
                model.load_state_dict(state_dict)
            else:
                raise ValueError("Could not determine model configuration from checkpoint")
                
        model.to(device)
        model.eval()
        return model
