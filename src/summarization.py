"""Multi-style text summarization examples."""

from transformers import pipeline
import torch
from config import DEVICE


def run_summarization_examples():
    """Run text summarization examples with different styles."""
    print("Initializing summarization pipeline...")
    
    # Use a smaller summarization model for better performance
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",  # Smaller distilled version
        device=0 if DEVICE == "cuda" else -1
    )
    
    # For style-based summarization, we'll also use a text generation model
    text_gen = pipeline(
        "text-generation",
        model="gpt2",
        device=0 if DEVICE == "cuda" else -1
    )
    
    # Sample business article
    article = """
    Apple reported record-breaking Q4 2024 earnings with revenue of $123.9 billion, 
    up 8% year-over-year. The company's services division showed particularly strong 
    growth at 12%, while iPhone sales remained stable. CEO Tim Cook highlighted the 
    successful launch of the iPhone 15 Pro and growing adoption of Apple Intelligence 
    features. The company also announced a $110 billion share buyback program and 
    increased its dividend by 4%. Looking forward, Apple guided for continued growth 
    in the services sector but warned of potential headwinds in the China market due 
    to increased competition from local manufacturers.
    """
    
    # Example 1: Standard summarization
    print("\n1. STANDARD SUMMARIZATION")
    print("-" * 50)
    print("Original article:", article[:100] + "...")
    
    summary = summarizer(article, max_length=60, min_length=30, do_sample=False)
    print(f"\nStandard summary: {summary[0]['summary_text']}")
    
    # Example 2: Multi-style summarization using prompts
    print("\n\n2. MULTI-STYLE SUMMARIZATION")
    print("-" * 50)
    
    prompts = {
        "executive": """You are an executive assistant. Provide a 2-sentence executive summary 
focusing on key financial metrics and strategic implications:

{text}

Executive Summary:""",
        
        "investor": """You are a financial analyst. Summarize for investors, highlighting:
- Revenue and growth figures
- Key business segments performance  
- Forward guidance and risks

Text: {text}

Investor Summary:""",
        
        "technical": """You are a tech journalist. Summarize focusing on:
- Product launches and adoption
- Technology innovations mentioned
- Competitive landscape

Text: {text}

Tech Summary:"""
    }
    
    for audience, prompt_template in prompts.items():
        prompt = prompt_template.format(text=article)
        response = text_gen(
            prompt,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=text_gen.tokenizer.eos_token_id
        )
        
        # Extract the summary part
        full_text = response[0]['generated_text']
        if "Summary:" in full_text:
            summary_text = full_text.split("Summary:")[-1].strip()
        else:
            summary_text = full_text[len(prompt):].strip()
            
        print(f"\n{audience.upper()} SUMMARY:")
        print(summary_text)
    
    # Example 3: Length-controlled summarization
    print("\n\n3. LENGTH-CONTROLLED SUMMARIZATION")
    print("-" * 50)
    
    lengths = [
        ("Tweet (280 chars)", 50),
        ("One-liner", 20),
        ("Paragraph", 100)
    ]
    
    for name, max_len in lengths:
        summary = summarizer(
            article,
            max_length=max_len,
            min_length=max_len // 2,
            do_sample=False
        )
        print(f"\n{name}:")
        print(summary[0]['summary_text'])
    
    # Example 4: Extractive vs Abstractive comparison
    print("\n\n4. EXTRACTIVE VS ABSTRACTIVE SUMMARIZATION")
    print("-" * 50)
    
    # Extractive-style (selecting key sentences)
    extractive_prompt = """Extract the 3 most important sentences from this text:

{text}

Important sentences:
1."""
    
    response = text_gen(
        extractive_prompt.format(text=article),
        max_new_tokens=150,
        temperature=0.3,
        do_sample=True,
        pad_token_id=text_gen.tokenizer.eos_token_id
    )
    print("Extractive-style summary:")
    print(response[0]['generated_text'].split("Important sentences:\n1.")[-1])
    
    # Abstractive (already shown above with BART)
    print("\nAbstractive summary (BART):")
    print(summary[0]['summary_text'])
    
    print("\n" + "=" * 50)
    print("Summarization examples completed!")


if __name__ == "__main__":
    run_summarization_examples()