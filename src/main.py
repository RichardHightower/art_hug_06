"""Main entry point for all examples."""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from named_entity_recognition import run_named_entity_recognition_examples
from question_answering import run_question_answering_examples
from text_generation import run_text_generation_examples
from multi_task_learning import run_multi_task_learning_examples
from summarization import run_summarization_examples
from conversational_ai import run_conversational_ai_examples
from document_processor import demo_document_processing
from prompt_manager import demo_prompt_manager
from secure_prompt import demo_secure_prompts

def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")

def main():
    """Run all examples."""
    print_section("CHAPTER 06: PROMPT ENGINEERING WITH TRANSFORMERS")
    print("Welcome! This script demonstrates prompt engineering concepts.")
    print("Each example builds on the previous concepts.\n")
    
    print_section("1. NAMED ENTITY RECOGNITION")
    run_named_entity_recognition_examples()
    
    print_section("2. TEXT GENERATION")
    run_text_generation_examples()
    
    print_section("3. QUESTION ANSWERING")
    run_question_answering_examples()
    
    print_section("4. TEXT SUMMARIZATION")
    run_summarization_examples()
    
    print_section("5. CONVERSATIONAL AI")
    run_conversational_ai_examples()
    
    print_section("6. DOCUMENT PROCESSING")
    demo_document_processing()
    
    print_section("7. PROMPT MANAGEMENT")
    demo_prompt_manager()
    
    print_section("8. SECURE PROMPTS")
    demo_secure_prompts()
    
    print_section("9. MULTI-TASK LEARNING")
    run_multi_task_learning_examples()
    
    print_section("CONCLUSION")
    print("These examples demonstrate key prompt engineering concepts.")
    print("Try modifying the code to experiment with different approaches!")

if __name__ == "__main__":
    main()
