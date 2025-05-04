# CSM MoE Optimization - Hackathon Roadmap

## Team Members
- **Rohan**: Lead developer, responsible for core implementation
- **Jonathan**: Documentation and analysis support

## Project Overview
Optimize Sesame CSM speech-to-speech system using a Mixture of Experts (MoE) architecture inspired by Llama 4. The goal is to improve memory usage and inference latency while maintaining output quality.

## Implementation Approach
We're using a custom vectorized MoE implementation that focuses on efficient parallel processing of tokens through experts. Each expert specializes in different aspects of speech pattern generation.

## Timeline
Remaining time: ~12 hours (May 3 evening - May 4 noon)

## Roadmap

### Phase 1: Integration (2 hours) - Rohan
- [ ] Integrate `VectorizedMoELayer` into CSM model architecture
- [ ] Implement CSM specific block wrapping in `csm_moe_block.py`
- [ ] Update `run_csm_moe.py` to use our optimized implementation
- [ ] Test basic functionality with a simple prompt

### Phase 1: Documentation & Support (2 hours) - Jonathan
- [ ] Update `README_MOE.md` with our implementation details
- [ ] Create a visualization template for expert specialization
- [ ] Prepare test prompts that represent different speech patterns
- [ ] Set up a spreadsheet to track performance metrics

### Phase 2: Performance Optimization (3 hours) - Rohan
- [ ] Implement mixed precision support for faster inference
- [ ] Optimize routing algorithm for better expert utilization
- [ ] Experiment with different number of experts (4, 8, 16)
- [ ] Implement memory profiling to track usage

```python
# Memory profiling code
def log_memory_stats(name):
    """Log memory stats at a specific point"""
    allocated = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    print(f"{name} - Allocated: {allocated:.2f}MB, Reserved: {reserved:.2f}MB")
```

### Phase 2: Testing & Analysis (3 hours) - Jonathan
- [ ] Run benchmarks with different configurations
- [ ] Track expert utilization patterns for different speech types
- [ ] Create visualizations of expert specialization
- [ ] Document memory usage improvements

```python
# Expert usage visualization example
def visualize_expert_usage(expert_counts, config_name):
    """Create a bar chart of expert usage"""
    import matplotlib.pyplot as plt
    
    total = sum(expert_counts)
    percentages = [count/total*100 for count in expert_counts]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(percentages)), percentages)
    plt.xlabel('Expert Index')
    plt.ylabel('Activation Percentage (%)')
    plt.title(f'Expert Utilization - {config_name}')
    plt.savefig(f'expert_usage_{config_name}.png')
```

### Phase 3: Benchmarking & Fine-tuning (2 hours) - Rohan
- [ ] Create comprehensive benchmarking system
- [ ] Compare vanilla CSM with different MoE configurations
- [ ] Fine-tune routing mechanisms based on speech patterns
- [ ] Document speed and memory improvements

### Phase 3: Presentation Preparation (2 hours) - Jonathan
- [ ] Create presentation slides
- [ ] Prepare demo script with clear speech examples
- [ ] Document business impact and technical achievements
- [ ] Create comparison charts showing performance gains

### Phase 4: Final Integration & Demo (1 hour) - Rohan & Jonathan
- [ ] Final code cleanup and integration
- [ ] Push to GitHub
- [ ] Prepare live demo for judges
- [ ] Practice presentation

## Technical Details

### Key Components
1. **VectorizedMoELayer**: Core MoE implementation with parallel processing
2. **OptimizedMoEBlockWrapper**: Wraps decoder blocks to add MoE capabilities
3. **CSMMoEModel**: Wraps the CSM model to enable MoE routing

### Implementation Strategy
- Replace decoder MLP components with MoE layers
- Implement top-k routing (k=2) to select most relevant experts
- Use vectorized operations for efficient processing
- Add tracking for expert utilization

### Configuration Options
- **num_experts**: Number of expert modules (4, 8, 16)
- **top_k**: Number of experts to route each token to (1, 2)
- **moe_blocks**: Which decoder blocks to apply MoE to (e.g., [0, 1])
- **mixed_precision**: Whether to use FP16 for faster inference

## Communication Protocol
- Check in every 30-45 minutes
- Use this roadmap to track progress
- Document all findings, even negative results

## Success Metrics
- Memory usage reduction (target: 20% reduction)
- Expert specialization (different experts handle different speech patterns)
- Quality preservation (audio quality remains high)
- Performance improvement (target: maintain or improve inference speed)

Let's build something awesome!
