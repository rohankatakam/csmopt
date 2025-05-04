import subprocess
import itertools
import json
import os
import sys
import argparse
from datetime import datetime

# Assuming logging_config.py is in the parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from logging_config import setup_logger

logger = setup_logger('experiment_runner')

def run_experiment(config, base_output_dir, train_script_path, pairs_path):
    \"\"\"Runs a single training experiment.\"\"\"
    exp_id = config["id"]
    exp_output_dir = os.path.join(base_output_dir, exp_id)
    os.makedirs(exp_output_dir, exist_ok=True)

    log_file = os.path.join(exp_output_dir, "train_adapter.log") # Log specific to this run

    logger.info(f"--- Running Experiment: {exp_id} --- Config: {config}")

    # Construct training command
    cmd = [
        sys.executable, # Use the same python interpreter
        train_script_path,
        "--pairs", pairs_path,
        "--output_dir", exp_output_dir,
        "--epochs", str(config["epochs"]),
        "--batch_size", str(config["batch_size"]),
        "--lr", str(config["lr"]),
        "--weight_decay", str(config["weight_decay"]),
        "--cosine_weight", str(config["cosine_weight"]),
        # Add other parameters if needed
    ]

    logger.info(f"Executing: {' '.join(cmd)}")

    try:
        # Run training, redirect output to log file and console
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        with open(log_file, 'w') as f:
            for line in iter(process.stdout.readline, ''):
                print(line, end='') # Print to console
                f.write(line)       # Write to file

        process.wait()
        return_code = process.returncode

        if return_code == 0:
            logger.info(f"Experiment {exp_id} completed successfully.")
            # Here you could parse the log file or look for checkpoint files
            # to extract final metrics if needed.
            result = {"status": "success", "output_dir": exp_output_dir}
        else:
            logger.error(f"Experiment {exp_id} failed with return code {return_code}.")
            result = {"status": "failed", "output_dir": exp_output_dir, "return_code": return_code}

    except Exception as e:
        logger.exception(f"Experiment {exp_id} failed with exception: {e}")
        result = {"status": "exception", "error": str(e)}

    return result

def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep for Llama4 Adapter training.")
    parser.add_argument("--pairs", type=str, required=True, help="Path to the training pairs.pt file.")
    parser.add_argument("--base_output_dir", type=str, default="./adapter_experiments",
                        help="Base directory to save experiment results.")
    args = parser.parse_args()

    # --- Define Parameter Space --- 
    # Modify these lists/values to define your sweep
    param_grid = {
        'lr': [1e-4, 5e-5, 1e-5],
        'batch_size': [16, 32],
        'epochs': [3, 5],
        'weight_decay': [0.01, 0.0],
        'cosine_weight': [0.3, 0.5, 0.7]
    }

    # Get script paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.abspath(os.path.join(script_dir, '..', 'train_adapter.py'))

    if not os.path.exists(train_script):
        logger.error(f"Training script not found at: {train_script}")
        sys.exit(1)

    if not os.path.exists(args.pairs):
        logger.error(f"Training pairs file not found at: {args.pairs}")
        sys.exit(1)

    # Create base output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = os.path.join(args.base_output_dir, f"run_{timestamp}")
    os.makedirs(run_output_dir, exist_ok=True)
    logger.info(f"Saving experiment results to: {run_output_dir}")

    # Generate all combinations of parameters
    keys, values = zip(*param_grid.items())
    experiment_configs = []
    for i, combo in enumerate(itertools.product(*values)):
        config = dict(zip(keys, combo))
        config['id'] = f"exp_{i:03d}" # Assign a unique ID
        experiment_configs.append(config)

    logger.info(f"Generated {len(experiment_configs)} experiment configurations.")

    # --- Run Experiments --- 
    results_summary = []
    for config in experiment_configs:
        result = run_experiment(config, run_output_dir, train_script, args.pairs)
        results_summary.append({"config": config, "result": result})

    # --- Save Summary --- 
    summary_file = os.path.join(run_output_dir, "summary_results.json")
    try:
        with open(summary_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        logger.info(f"Experiment summary saved to {summary_file}")
    except Exception as e:
        logger.exception(f"Failed to save summary file: {e}")

    logger.info("--- Experiment run finished. ---")

if __name__ == "__main__":
    main() 