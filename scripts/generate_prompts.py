import argparse
import random
import sys
import os

# Assuming logging_config.py is in the parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from logging_config import setup_logger
    logger = setup_logger('prompt_generator')
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('prompt_generator')
    logger.warning("Could not import logging_config. Using basic logging.")


# --- List of Seed Prompts ---
# Expand this list with diverse, relevant prompts for your target use case
SEED_PROMPTS = [
    # Simple questions
    "Hello, how are you today?",
    "What is the weather like?",
    "Can you tell me a joke?",
    "What time is it?",
    "Explain the concept of gravity.",
    "Who was Albert Einstein?",

    # Simple instructions
    "Set a timer for 5 minutes.",
    "Add milk to my shopping list.",
    "Play some relaxing music.",
    "Turn off the living room lights.",
    "Tell me the news headlines.",
    "Send a message to John saying I'll be late.",

    # Conversational starters
    "What are your capabilities?",
    "Tell me something interesting.",
    "How does speech synthesis work?",
    "What's new in the world of AI?",
    "Can you help me brainstorm ideas for a story?",
    "Let's chat about technology.",

    # More specific (examples, adapt to your needs)
    "Book a table for two at a nearby Italian restaurant for 7 PM tonight.",
    "What are the side effects of paracetamol?",
    "Give me directions to the nearest post office.",
    "Summarize the plot of the movie Inception.",
    "Translate 'hello world' into French.",
    "Write a short poem about the rain.",
]

def generate_prompts(num_prompts: int) -> list[str]:
    \"\"\"Generates a list of prompts by randomly choosing from seeds.\"\"\"
    # Simple strategy: just sample with replacement
    # More complex strategies could involve templates, variations, etc.
    if not SEED_PROMPTS:
        logger.error("SEED_PROMPTS list is empty. Cannot generate prompts.")
        return []
    
    generated = [random.choice(SEED_PROMPTS) for _ in range(num_prompts)]
    logger.info(f"Generated {len(generated)} prompts.")
    return generated

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic prompts for training data.")
    parser.add_argument("--num_prompts", type=int, default=5000, 
                        help="Number of prompts to generate.")
    parser.add_argument("--output_file", type=str, default="prompts.txt",
                        help="File to save the generated prompts (one per line).")
    args = parser.parse_args()

    if args.num_prompts <= 0:
        logger.error("Number of prompts must be positive.")
        return

    logger.info(f"Generating {args.num_prompts} prompts...")
    prompts = generate_prompts(args.num_prompts)

    if prompts:
        try:
            with open(args.output_file, 'w') as f:
                for prompt in prompts:
                    f.write(prompt + '\n')
            logger.info(f"Successfully saved prompts to {args.output_file}")
        except IOError as e:
            logger.error(f"Failed to write prompts to {args.output_file}: {e}")

if __name__ == "__main__":
    main() 