# CSM MoE Optimization Layer

This repository implements a lightweight adaptive routing mechanism inspired by LLaMA 4's Mixture of Experts (MoE) architecture for the [Sesame CSM](https://github.com/SesameAILabs/csm) speech-to-speech system.

## Overview

The MoE routing mechanism dynamically activates only the most relevant expert layers in CSM's decoder, resulting in:

- Improved memory usage during inference
- Reduced inference latency
- Maintained or enhanced output quality

## Architecture

The implementation consists of:

1. **`moe_router.py`**: Core MoE implementation with top-k gating mechanism
   - `MoEGating`: Top-k gating module that determines which experts to use
   - `MoEExpertLayer`: Individual expert implementation (MLP-based)
   - `MoELayer`: Full MoE layer with router and multiple experts
   - `MoEBlockWrapper`: Wrapper for integrating MoE with existing decoder blocks
   - Utility functions for patching LLaMA blocks with MoE

2. **`csm_moe_block.py`**: Integration with CSM architecture
   - `CSMMoEModel`: Wrapper for the CSM model with MoE capabilities
   - Helper functions for creating MoE-enabled CSM models

3. **`run_csm_moe.py`**: Test script for running and benchmarking MoE routing

## How it Works

The MoE implementation follows the LLaMA 4 approach:

1. A router network examines each input token and assigns it to the top-k most relevant experts (default k=2)
2. Each selected expert processes the token
3. The outputs from the selected experts are weighted and combined
4. The MoE layers replace the MLP components in selected decoder blocks

## Usage

To run CSM with MoE routing:

```bash
python run_csm_moe.py [options]
```

### Options

- `--moe_blocks`: Comma-separated list of decoder block indices to apply MoE (default: `0,1`)
- `--num_experts`: Number of experts in MoE layers (default: `8`)
- `--top_k`: Number of experts to route to (default: `2`)
- `--disable_moe`: Disable MoE routing for comparison
- `--benchmark`: Run in benchmark mode to compare performance
- `--output`: Output WAV file name (default: `moe_conversation.wav`)

### Example

```bash
# Run with MoE on decoder blocks 0 and 1, using 8 experts and top-2 routing
python run_csm_moe.py --moe_blocks=0,1 --num_experts=8 --top_k=2

# Run benchmark to compare performance with and without MoE
python run_csm_moe.py --benchmark

# Disable MoE for baseline comparison
python run_csm_moe.py --disable_moe
```

## Benchmarking

The benchmark mode compares generation time with and without MoE routing.
It also provides statistics on expert usage to analyze routing patterns.

## Future Improvements

Potential enhancements:

1. Implement more sophisticated load balancing for better expert utilization
2. Add auxiliary loss components from the Switch Transformers paper
3. Optimize MoE dispatch for larger batch sizes
4. Add support for different expert architectures beyond MLPs

## References

- [Sesame CSM paper](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice)
- [LLaMA 4 blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)
- [Switch Transformers paper](https://arxiv.org/abs/2101.03961)
