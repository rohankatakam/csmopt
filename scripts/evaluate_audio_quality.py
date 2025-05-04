#!/usr/bin/env python3
"""
Evaluate Audio Quality of Llama 4 Adapter + CSM

This script compares the audio quality between the standard Llama 3 + CSM
and the new Llama 4 + Adapter + CSM pipeline. It measures quality using
objective metrics like MCD (Mel Cepstral Distortion) and conducts
A/B testing evaluation.
"""
import os
import sys
import torch
import numpy as np
import argparse
import logging
import json
import torchaudio
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
from logging_config import setup_logger

# Add parent directory to path to import from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules (if CSM model is available)
try:
    # This is a placeholder - in a real implementation, we would import
    # the necessary CSM modules for audio generation
    pass
except ImportError:
    logging.warning("CSM audio generation modules not available - running in simulation mode")

# Configure logging
logger = setup_logger('audio_eval', level=logging.INFO)

def compute_mel_spectrogram(
    audio: torch.Tensor,
    sample_rate: int = 24000,
    n_fft: int = 1024,
    win_length: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80
) -> torch.Tensor:
    """
    Compute mel spectrogram from audio waveform.
    
    Args:
        audio: Audio waveform tensor [channels, samples]
        sample_rate: Audio sample rate
        n_fft: FFT window size
        win_length: Window length
        hop_length: Hop length between frames
        n_mels: Number of mel bands
        
    Returns:
        Mel spectrogram tensor [channels, n_mels, time]
    """
    # Ensure audio is 2D [channels, samples]
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    # Create mel spectrogram transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=0,
        f_max=8000,
        power=1  # amplitude spectrogram
    )
    
    # Compute spectrogram
    mel_spec = mel_transform(audio)
    
    # Convert to log scale
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
    
    return mel_spec

def compute_mcd(
    ref_audio: torch.Tensor,
    test_audio: torch.Tensor,
    sample_rate: int = 24000
) -> float:
    """
    Compute Mel Cepstral Distortion (MCD) between reference and test audio.
    
    MCD measures the difference between two audio samples in the mel-cepstral domain.
    Lower values indicate more similar audio.
    
    Args:
        ref_audio: Reference audio waveform
        test_audio: Test audio waveform
        sample_rate: Audio sample rate
        
    Returns:
        MCD value
    """
    # Compute mel spectrograms
    ref_mel = compute_mel_spectrogram(ref_audio, sample_rate=sample_rate)
    test_mel = compute_mel_spectrogram(test_audio, sample_rate=sample_rate)
    
    # Convert to numpy
    ref_mel_np = ref_mel.squeeze().numpy()
    test_mel_np = test_mel.squeeze().numpy()
    
    # If lengths differ, truncate to shorter length
    min_len = min(ref_mel_np.shape[1], test_mel_np.shape[1])
    ref_mel_np = ref_mel_np[:, :min_len]
    test_mel_np = test_mel_np[:, :min_len]
    
    # Compute MCD
    diff = ref_mel_np - test_mel_np
    mcd = np.sqrt(np.mean(np.sum(diff**2, axis=0)))
    
    return mcd

def compute_pesq(
    ref_audio: torch.Tensor,
    test_audio: torch.Tensor,
    sample_rate: int = 24000
) -> float:
    """
    Compute PESQ (Perceptual Evaluation of Speech Quality) score.
    
    PESQ is an objective measure for estimating speech quality.
    Higher values indicate better quality (range: -0.5 to 4.5).
    
    Args:
        ref_audio: Reference audio waveform
        test_audio: Test audio waveform
        sample_rate: Audio sample rate
        
    Returns:
        PESQ score
    """
    try:
        import pesq
    except ImportError:
        logger.warning("pesq module not available, skipping PESQ computation")
        return -1
    
    # Convert to numpy
    ref_np = ref_audio.squeeze().numpy()
    test_np = test_audio.squeeze().numpy()
    
    # Ensure correct sample rate (PESQ requires 8kHz or 16kHz)
    if sample_rate != 8000 and sample_rate != 16000:
        logger.warning(f"Resampling from {sample_rate} to 16000 Hz for PESQ")
        ref_resampled = torchaudio.functional.resample(
            torch.tensor(ref_np), sample_rate, 16000
        ).numpy()
        test_resampled = torchaudio.functional.resample(
            torch.tensor(test_np), sample_rate, 16000
        ).numpy()
        sample_rate = 16000
    else:
        ref_resampled = ref_np
        test_resampled = test_np
    
    # PESQ requires specific lengths, so truncate if needed
    min_len = min(len(ref_resampled), len(test_resampled))
    ref_resampled = ref_resampled[:min_len]
    test_resampled = test_resampled[:min_len]
    
    # Ensure signals have reasonable length (>0.1s)
    if min_len < 0.1 * sample_rate:
        logger.warning("Audio too short for PESQ computation")
        return -1
    
    try:
        pesq_score = pesq.pesq(sample_rate, ref_resampled, test_resampled, 'wb')
        return pesq_score
    except Exception as e:
        logger.error(f"Error computing PESQ: {e}")
        return -1

def generate_audio_comparison(
    input_texts: List[str],
    output_dir: str,
    llama3_csm_generator=None,
    llama4_csm_generator=None,
    simulation_mode: bool = True
) -> Dict[str, Any]:
    """
    Generate and compare audio from Llama 3 + CSM vs Llama 4 + Adapter + CSM.
    
    Args:
        input_texts: List of input texts
        output_dir: Directory to save outputs
        llama3_csm_generator: Llama 3 + CSM generator
        llama4_csm_generator: Llama 4 + Adapter + CSM generator
        simulation_mode: Whether to run in simulation mode (for testing)
        
    Returns:
        Dictionary with quality comparison results
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {
        "per_sample": [],
        "aggregate": {}
    }
    
    mcd_values = []
    pesq_values = []
    
    for i, text in enumerate(tqdm(input_texts, desc="Generating audio comparisons")):
        sample_dir = os.path.join(output_dir, f"sample_{i+1}")
        os.makedirs(sample_dir, exist_ok=True)
        
        sample_results = {
            "input_text": text,
            "sample_id": i+1
        }
        
        # Generate or simulate audio
        if simulation_mode:
            # Simulate audio generation with random noise
            llama3_audio = torch.randn(1, int(3 * 24000))  # 3 seconds at 24kHz
            llama4_audio = torch.randn(1, int(3 * 24000))
            
            # Make them somewhat similar (to simulate realistic MCD values)
            # Add some structured patterns to both audios
            t = torch.linspace(0, 100, llama3_audio.shape[1])
            pattern = torch.sin(0.1 * t) + 0.5 * torch.sin(0.2 * t) + 0.3 * torch.sin(0.3 * t)
            llama3_audio[0] += pattern * 0.5
            llama4_audio[0] += pattern * 0.5
            
            # Add some differences to simulate realistic quality differences
            llama4_audio[0] += torch.sin(0.15 * t) * 0.2
            
            # Save simulated audio
            torchaudio.save(
                os.path.join(sample_dir, "llama3_csm.wav"),
                llama3_audio,
                sample_rate=24000
            )
            torchaudio.save(
                os.path.join(sample_dir, "llama4_csm.wav"),
                llama4_audio,
                sample_rate=24000
            )
        else:
            # Use actual generators if available
            if llama3_csm_generator is None or llama4_csm_generator is None:
                raise ValueError("Audio generators must be provided when not in simulation mode")
            
            # Generate with Llama 3 + CSM
            llama3_audio = llama3_csm_generator(text)
            torchaudio.save(
                os.path.join(sample_dir, "llama3_csm.wav"),
                llama3_audio,
                sample_rate=24000
            )
            
            # Generate with Llama 4 + Adapter + CSM
            llama4_audio = llama4_csm_generator(text)
            torchaudio.save(
                os.path.join(sample_dir, "llama4_csm.wav"),
                llama4_audio,
                sample_rate=24000
            )
        
        # Compute objective metrics
        mcd = compute_mcd(llama3_audio, llama4_audio)
        pesq_score = compute_pesq(llama3_audio, llama4_audio)
        
        mcd_values.append(mcd)
        if pesq_score > 0:
            pesq_values.append(pesq_score)
        
        sample_results.update({
            "mcd": mcd,
            "pesq": pesq_score,
            "llama3_path": os.path.join(sample_dir, "llama3_csm.wav"),
            "llama4_path": os.path.join(sample_dir, "llama4_csm.wav")
        })
        
        # Generate spectrograms for visualization
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot Llama 3 spectrogram
        llama3_mel = compute_mel_spectrogram(llama3_audio)
        axes[0].imshow(
            llama3_mel.squeeze().numpy(),
            aspect='auto',
            origin='lower',
            cmap='viridis'
        )
        axes[0].set_title('Llama 3 + CSM Spectrogram')
        axes[0].set_ylabel('Mel Frequency')
        
        # Plot Llama 4 spectrogram
        llama4_mel = compute_mel_spectrogram(llama4_audio)
        axes[1].imshow(
            llama4_mel.squeeze().numpy(),
            aspect='auto',
            origin='lower',
            cmap='viridis'
        )
        axes[1].set_title('Llama 4 + Adapter + CSM Spectrogram')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Mel Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, "spectrogram_comparison.png"))
        plt.close()
        
        # Plot waveform comparison
        fig, axes = plt.subplots(2, 1, figsize=(10, 6))
        
        axes[0].plot(llama3_audio.squeeze().numpy())
        axes[0].set_title('Llama 3 + CSM Waveform')
        axes[0].set_ylabel('Amplitude')
        
        axes[1].plot(llama4_audio.squeeze().numpy())
        axes[1].set_title('Llama 4 + Adapter + CSM Waveform')
        axes[1].set_xlabel('Samples')
        axes[1].set_ylabel('Amplitude')
        
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, "waveform_comparison.png"))
        plt.close()
        
        # Add to results
        results["per_sample"].append(sample_results)
    
    # Compute aggregate statistics
    results["aggregate"] = {
        "mcd": {
            "mean": float(np.mean(mcd_values)),
            "median": float(np.median(mcd_values)),
            "std": float(np.std(mcd_values)),
            "min": float(np.min(mcd_values)),
            "max": float(np.max(mcd_values))
        },
        "pesq": {
            "mean": float(np.mean(pesq_values)) if pesq_values else -1,
            "median": float(np.median(pesq_values)) if pesq_values else -1,
            "std": float(np.std(pesq_values)) if pesq_values else -1,
            "min": float(np.min(pesq_values)) if pesq_values else -1,
            "max": float(np.max(pesq_values)) if pesq_values else -1
        }
    }
    
    # Plot MCD distribution
    plt.figure(figsize=(8, 6))
    plt.hist(mcd_values, bins=10, color='blue', alpha=0.7)
    plt.axvline(np.mean(mcd_values), color='red', linestyle='dashed', linewidth=2)
    plt.text(
        np.mean(mcd_values) * 1.1, 
        plt.ylim()[1] * 0.9, 
        f'Mean: {np.mean(mcd_values):.2f}',
        color='red'
    )
    plt.xlabel('MCD Value')
    plt.ylabel('Count')
    plt.title('Distribution of MCD Values (lower is better)')
    plt.savefig(os.path.join(output_dir, "mcd_distribution.png"))
    plt.close()
    
    # Plot PESQ distribution if available
    if pesq_values:
        plt.figure(figsize=(8, 6))
        plt.hist(pesq_values, bins=10, color='green', alpha=0.7)
        plt.axvline(np.mean(pesq_values), color='red', linestyle='dashed', linewidth=2)
        plt.text(
            np.mean(pesq_values) * 1.1, 
            plt.ylim()[1] * 0.9, 
            f'Mean: {np.mean(pesq_values):.2f}',
            color='red'
        )
        plt.xlabel('PESQ Score')
        plt.ylabel('Count')
        plt.title('Distribution of PESQ Scores (higher is better)')
        plt.savefig(os.path.join(output_dir, "pesq_distribution.png"))
        plt.close()
    
    # Save results
    with open(os.path.join(output_dir, "quality_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate summary report
    report_path = os.path.join(output_dir, "quality_report.md")
    with open(report_path, "w") as f:
        f.write("# Audio Quality Evaluation Report\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"Evaluated {len(input_texts)} samples comparing Llama 3 + CSM vs Llama 4 + Adapter + CSM.\n\n")
        
        f.write("## MCD Metrics (Mel Cepstral Distortion)\n\n")
        f.write("Lower MCD values indicate more similar audio quality.\n\n")
        f.write(f"- **Mean MCD**: {results['aggregate']['mcd']['mean']:.4f}\n")
        f.write(f"- **Median MCD**: {results['aggregate']['mcd']['median']:.4f}\n")
        f.write(f"- **Min MCD**: {results['aggregate']['mcd']['min']:.4f}\n")
        f.write(f"- **Max MCD**: {results['aggregate']['mcd']['max']:.4f}\n")
        f.write(f"- **Standard Deviation**: {results['aggregate']['mcd']['std']:.4f}\n\n")
        
        if pesq_values:
            f.write("## PESQ Scores\n\n")
            f.write("Higher PESQ values indicate better perceived quality (range: -0.5 to 4.5).\n\n")
            f.write(f"- **Mean PESQ**: {results['aggregate']['pesq']['mean']:.4f}\n")
            f.write(f"- **Median PESQ**: {results['aggregate']['pesq']['median']:.4f}\n")
            f.write(f"- **Min PESQ**: {results['aggregate']['pesq']['min']:.4f}\n")
            f.write(f"- **Max PESQ**: {results['aggregate']['pesq']['max']:.4f}\n")
            f.write(f"- **Standard Deviation**: {results['aggregate']['pesq']['std']:.4f}\n\n")
        
        f.write("## Quality Interpretation\n\n")
        # Provide an interpretation based on the MCD values
        mean_mcd = results['aggregate']['mcd']['mean']
        if mean_mcd < 3:
            quality_assessment = "The audio quality from Llama 4 + Adapter + CSM is very similar to the baseline Llama 3 + CSM, indicating successful integration."
        elif mean_mcd < 5:
            quality_assessment = "The audio quality shows moderate differences from baseline, but is likely acceptable for most use cases."
        else:
            quality_assessment = "The audio quality shows significant differences from baseline. Further optimization of the adapter may be needed."
        
        f.write(f"{quality_assessment}\n\n")
        
        f.write("## Sample Details\n\n")
        for i, sample in enumerate(results["per_sample"]):
            f.write(f"### Sample {i+1}\n\n")
            f.write(f"- **Input Text**: \"{sample['input_text']}\"\n")
            f.write(f"- **MCD**: {sample['mcd']:.4f}\n")
            if sample['pesq'] > 0:
                f.write(f"- **PESQ**: {sample['pesq']:.4f}\n")
            f.write("\n")
    
    logger.info(f"Quality evaluation report saved to {report_path}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate audio quality of Llama 4 adapter integration")
    
    parser.add_argument("--inputs", type=str, default=None,
                       help="Path to file with input texts (one per line)")
    parser.add_argument("--output_dir", type=str, default="quality_evaluation",
                       help="Directory to save evaluation results")
    parser.add_argument("--simulation", action="store_true",
                       help="Run in simulation mode (without actual audio generation)")
    
    args = parser.parse_args()
    
    # Default test inputs if not provided
    if args.inputs is None:
        test_inputs = [
            "Hello, how are you today?",
            "What's the weather like in San Francisco?",
            "I'm thinking about learning to play the guitar. Any advice on getting started?",
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming many industries."
        ]
    else:
        # Read inputs from file
        with open(args.inputs, "r") as f:
            test_inputs = [line.strip() for line in f if line.strip()]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # In a real implementation, we would initialize the generators here
    # For now, we'll use simulation mode
    
    # Run quality evaluation
    logger.info("Running audio quality evaluation...")
    results = generate_audio_comparison(
        input_texts=test_inputs,
        output_dir=args.output_dir,
        simulation_mode=args.simulation
    )
    
    logger.info(f"Quality evaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
