"""Simplified main entry point that runs a subset of examples."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from named_entity_recognition import run_named_entity_recognition_examples
from text_generation import run_text_generation_examples
from summarization import run_summarization_examples

def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")

def main():
    """Run key examples."""
    print_section("PROMPT ENGINEERING WITH TRANSFORMERS - KEY EXAMPLES")
    print("Running a selection of prompt engineering examples.\n")
    
    print_section("1. NAMED ENTITY RECOGNITION")
    run_named_entity_recognition_examples()
    
    print_section("2. TEXT GENERATION")
    run_text_generation_examples()
    
    print_section("3. TEXT SUMMARIZATION")
    run_summarization_examples()
    
    print_section("CONCLUSION")
    print("Examples completed! For more examples, see the Jupyter notebook.")
    print("Run 'task notebook' to explore interactive examples.")

if __name__ == "__main__":
    main()