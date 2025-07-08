"""Text generation examples using Hugging Face Transformers."""

from transformers import pipeline
import torch
from config import DEVICE, DEFAULT_MODEL


def run_text_generation_examples():
    """Run text generation examples from the article."""
    print("Initializing text generation pipeline...")
    
    # Use a smaller model for demonstration
    text_gen = pipeline(
        "text-generation",
        model="gpt2",  # Using GPT-2 as it's more accessible
        device=0 if DEVICE == "cuda" else -1
    )
    
    # Example 1: Comparing prompt variations
    print("\n1. COMPARING PROMPT VARIATIONS")
    print("-" * 50)
    
    prompts = [
        "Explain quantum computing in simple terms.",
        "Imagine you're teaching quantum computing to a 10-year-old. How would you explain it?",
        "As a science teacher, explain quantum computing to a 10-year-old, step by step."
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\nPrompt {i}: {prompt}")
        response = text_gen(
            prompt,
            max_new_tokens=30,
            temperature=0.8,
            do_sample=True,
            pad_token_id=text_gen.tokenizer.eos_token_id,
            truncation=True,
            max_length=100
        )
        print(f"Response: {response[0]['generated_text']}")
    
    # Example 2: Role prompting
    print("\n\n2. ROLE PROMPTING EXAMPLES")
    print("-" * 50)
    
    role_prompts = [
        "You are a science teacher. Explain how a neural network learns.",
        "You are a chef. Explain how a neural network learns using cooking analogies.",
        "You are a sports coach. Explain how a neural network learns using sports training analogies."
    ]
    
    for prompt in role_prompts:
        print(f"\nPrompt: {prompt}")
        response = text_gen(
            prompt,
            max_new_tokens=80,
            temperature=0.7,
            do_sample=True,
            pad_token_id=text_gen.tokenizer.eos_token_id
        )
        print(f"Response: {response[0]['generated_text']}")
    
    # Example 3: Chain-of-thought prompting
    print("\n\n3. CHAIN-OF-THOUGHT PROMPTING")
    print("-" * 50)
    
    cot_prompt = """Solve this step by step: If a train travels 60 miles per hour for 2.5 hours, how far does it travel?

Step 1: Identify what we know
Step 2: Apply the formula
Step 3: Calculate the answer

Let me solve this step by step:"""
    
    print(f"Prompt: {cot_prompt}")
    response = text_gen(
        cot_prompt,
        max_new_tokens=100,
        temperature=0.5,
        do_sample=True,
        pad_token_id=text_gen.tokenizer.eos_token_id
    )
    print(f"Response: {response[0]['generated_text']}")
    
    # Example 4: Creative text generation
    print("\n\n4. CREATIVE TEXT GENERATION")
    print("-" * 50)
    
    creative_prompts = [
        "Write a haiku about artificial intelligence:",
        "Complete this story: The robot opened its eyes for the first time and",
        "Generate a product description for an AI-powered coffee maker:"
    ]
    
    for prompt in creative_prompts:
        print(f"\nPrompt: {prompt}")
        response = text_gen(
            prompt,
            max_new_tokens=50,
            temperature=0.9,
            do_sample=True,
            pad_token_id=text_gen.tokenizer.eos_token_id
        )
        print(f"Response: {response[0]['generated_text']}")
    
    print("\n" + "=" * 50)
    print("Text generation examples completed!")


if __name__ == "__main__":
    run_text_generation_examples()