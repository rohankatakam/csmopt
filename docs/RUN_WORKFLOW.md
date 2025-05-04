# End-to-End Workflow: Llama-4 Adapter Training and Integration

This document outlines the steps required to generate training data, train the Llama-4 adapter, and use it for inference with the CSM decoder.

**Refer also to:** `docs/TRANSLATION_LAYER.md` for the original integration plan and technical details.

## Prerequisites

1.  **Python Environment:** Set up a Python environment (e.g., conda or venv) with Python 3.9+.
2.  **Dependencies:** Install required packages:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Model Access:**
    *   You need access to the gated Hugging Face models:
        *   `meta-llama/llama-4-scout-17b` (or your target Llama-4 variant)
        *   `meta-llama/Llama-3` (or your specific Llama-3 baseline model used in `dump_pairs.py`)
        *   `sesame/csm-1b` (or your specific CSM decoder model)
    *   Set your Hugging Face token as an environment variable:
        ```bash
        export HUGGING_FACE_HUB_TOKEN='your_hf_token_here'
        ```
    *   Alternatively, pass the token via script arguments where available.
4.  **Model Downloads:** Ensure the models are downloaded or accessible. The scripts will attempt to download them via `transformers` if not found locally, which requires sufficient disk space.
5.  **Hardware (CRITICAL):**
    *   **Data Generation (`scripts/dump_pairs.py`):** Requires loading **two** large language models (Llama-3 and Llama-4). This likely needs **> 32GB GPU VRAM**, potentially more depending on quantization.
    *   **Adapter Training (`train_adapter.py`, `scripts/run_experiments.py`):** Requires a high-end GPU, typically with **>= 24GB VRAM** (like an RTX 3090/4090 or A100), especially when processing the 8192-dimensional Llama-4 hidden states.
    *   **Inference (`scripts/run_csm_with_llama4.py`):** Requires loading Llama-4 and the adapter. **>= 24GB VRAM** is recommended for quantized models.
    *   **AWS Lambda is NOT suitable for the data generation or training steps due to these resource requirements.**

## Workflow Steps

### Step 1: Generate Prompts

Create a set of text prompts that will be used to generate the training data pairs.

*   **Script:** `scripts/generate_prompts.py`
*   **Purpose:** Creates a file containing text prompts (one per line).
*   **Recommendation:** Customize the `SEED_PROMPTS` list within the script to better match your target application's domain and style.

```bash
python scripts/generate_prompts.py \
    --num_prompts 5000 \
    --output_file data/training_prompts.txt
```

*   **Output:** `data/training_prompts.txt` (or your specified path).

### Step 2: Generate Training Data Pairs (`pairs.pt`)

Run the dual inference process: feed the prompts to both the Llama-3+CSM stack and the Llama-4 model. Capture the Llama-4 hidden states (`h4`) and the corresponding CSM decoder inputs/logits (`c`).

*   **Script:** `scripts/dump_pairs.py`
*   **Purpose:** Generates the core training data file `(h4, c)`.
*   **Hardware:** **Requires high-end GPU(s) with significant VRAM (>32GB recommended).**
*   **Note:** You **must** check the specific arguments required by `scripts/dump_pairs.py` using `python scripts/dump_pairs.py --help`. The command below is an example.

```bash
# Example - Verify arguments with --help
python scripts/dump_pairs.py \
    --prompts data/training_prompts.txt \
    --output data/adapter_training_pairs.pt \
    --llama3_model_path /path/or/name/to/llama3/baseline \
    --llama4_model_path meta-llama/llama-4-scout-17b \
    --csm_decoder_path /path/or/name/to/csm/decoder \
    --device cuda
```

*   **Output:** `data/adapter_training_pairs.pt` (or your specified path).
*   **Logs:** Check `logs/` directory for logs from this script (if logging is configured within it).

### Step 3: Train the Adapter

Train the lightweight MLP adapter to map Llama-4 hidden states to the CSM decoder's expected input dimension.

*   **Script:** `train_adapter.py`
*   **Purpose:** Trains and saves the adapter model checkpoint.
*   **Hardware:** **Requires a high-end GPU (>= 24GB VRAM recommended).**

**Option A: Single Training Run**

```bash
python train_adapter.py \
    --pairs data/adapter_training_pairs.pt \
    --output_dir checkpoints/adapter_run_01 \
    --epochs 3 \
    --batch_size 32 \
    --lr 5e-5 \
    --cosine_weight 0.5
```

*   **Output:** Checkpoints saved in `checkpoints/adapter_run_01`. Look for `adapter_only.pt` (recommended for inference) and potentially `adapter.ckpt` (full training state).
*   **Logs:** A timestamped log file (e.g., `logs/train_adapter_*.log`) will be created via `logging_config.py`.

**Option B: Hyperparameter Sweep**

Use the experiment runner to automatically train with multiple hyperparameter combinations.

*   **Script:** `scripts/run_experiments.py`
*   **Purpose:** Orchestrates multiple runs of `train_adapter.py`.
*   **Configuration:** Edit the `param_grid` dictionary within `scripts/run_experiments.py` to define the hyperparameters you want to test.

```bash
python scripts/run_experiments.py \
    --pairs data/adapter_training_pairs.pt \
    --base_output_dir results/adapter_experiments
```

*   **Output:**
    *   A new directory under `results/adapter_experiments/run_YYYYMMDD_HHMMSS/` containing subdirectories for each experiment (`exp_000`, `exp_001`, ...).
    *   Each `exp_...` directory contains checkpoints (`adapter_only.pt`) and logs (`train_adapter.log`) for that specific run.
    *   A `summary_results.json` file summarizing all configurations and their status.
*   **Logs:** The main runner log (`logs/experiment_runner_*.log`) and individual training logs within each experiment directory.

### Step 4: Evaluate and Select the Best Adapter (Recommended)

After training (especially after a hyperparameter sweep), evaluate the performance of different adapters to choose the best one.

*   **Script:** `scripts/evaluate_audio_quality.py` (or similar evaluation scripts you develop).
*   **Purpose:** Compares the audio output generated using different adapters against a baseline or reference audio.
*   **Input:** Requires evaluation prompts, the base models (Llama-4, CSM Decoder), and the paths to the different trained adapter checkpoints (`adapter_only.pt` from Step 3).

1.  Run your evaluation script, passing the necessary model/adapter paths.
2.  Analyze the output metrics (e.g., MCD, PESQ, MOS) saved by the evaluation script.
3.  Select the `adapter_only.pt` file from the experiment run that yielded the best evaluation results.

### Step 5: Run Inference with the Trained Adapter

Use the chosen adapter checkpoint in the final inference pipeline.

*   **Script:** `scripts/run_csm_with_llama4.py`
*   **Purpose:** Takes text input, uses Llama-4 + the selected adapter to generate hidden states, and (ideally) passes them to the CSM decoder to produce audio.
*   **Hardware:** **Requires GPU capable of running Llama-4 (>= 24GB VRAM recommended).**

```bash
# Use the path to the adapter selected in Step 4
python scripts/run_csm_with_llama4.py \
    --llama4_model meta-llama/llama-4-scout-17b \
    --adapter /path/to/your/best/adapter_only.pt \
    --csm_model /path/or/name/to/csm/decoder \
    --prompt "Generate a short audio clip saying hello world." \
    --output_wav output/hello_world.wav \
    --device cuda
```

*   **Output:** `output/hello_world.wav` (or your specified path).
*   **Logs:** Check `logs/run_csm_llama4_*.log`.

## Logging

Most scripts are configured to use the centralized `logging_config.py`. Logs are typically written to:

*   The console.
*   Timestamped files within the `logs/` directory (e.g., `logs/script_name_YYYYMMDD_HHMMSS.log`).
*   Experiment-specific log files (e.g., `results/adapter_experiments/run_.../exp_.../train_adapter.log`).

You can adjust the default log level (`INFO`) in the `setup_logger` calls within each script or via command-line arguments where available (e.g., `--verbose` in `scripts/run_csm_with_llama4.py`). 