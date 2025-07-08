"""Question answering examples with smart QA system implementation."""

from transformers import pipeline
import json
from typing import Dict, List
from config import DEVICE


class SmartQASystem:
    """Production-ready question answering system with confidence scoring."""
    
    def __init__(self, model=None):
        """Initialize the QA system with a text generation model."""
        if model is None:
            self.model = pipeline(
                "text-generation",
                model="gpt2",  # Using GPT-2 for accessibility
                device=0 if DEVICE == "cuda" else -1
            )
        else:
            self.model = model
            
        self.context_template = """You are a helpful AI assistant with expertise in {domain}.
        
Context: {context}

Question: {question}

Instructions:
1. Answer based ONLY on the provided context
2. If the answer isn't in the context, say "I don't have enough information"
3. Be concise but complete
4. Use bullet points for multiple items

Answer:"""
        
    def answer_with_confidence(self, question: str, context: str, domain: str = "general") -> Dict:
        """Answer a question with confidence scoring."""
        # First attempt: Direct answer
        prompt = self.context_template.format(
            domain=domain,
            context=context,
            question=question
        )
        
        response = self.model(
            prompt,
            max_new_tokens=200,
            temperature=0.3,  # Lower temperature for factual accuracy
            do_sample=True,
            pad_token_id=self.model.tokenizer.eos_token_id
        )
        
        # Extract answer after "Answer:"
        full_response = response[0]['generated_text']
        if "Answer:" in full_response:
            answer = full_response.split("Answer:")[-1].strip()
        else:
            answer = full_response[len(prompt):].strip()
        
        # Self-verification prompt
        verify_prompt = f"""Given this context: {context}
        
Question: {question}
Answer provided: {answer}

Is this answer accurate and complete based ONLY on the context? 
Respond with 'Yes' or 'No' and explain briefly."""
        
        verification = self.model(
            verify_prompt,
            max_new_tokens=50,
            temperature=0.3,
            do_sample=True,
            pad_token_id=self.model.tokenizer.eos_token_id
        )
        
        verification_text = verification[0]['generated_text']
        
        return {
            "answer": answer,
            "verification": verification_text,
            "confidence": "high" if "Yes" in verification_text else "low"
        }


def run_question_answering_examples():
    """Run question answering examples from the article."""
    print("Initializing Question Answering System...")
    qa_system = SmartQASystem()
    
    # Example 1: Company knowledge base
    print("\n1. COMPANY KNOWLEDGE BASE Q&A")
    print("-" * 50)
    
    context = """
TechCorp's new AI platform, CloudMind, offers three tiers:
- Starter: $99/month, 10,000 API calls, basic models
- Professional: $499/month, 100,000 API calls, advanced models, priority support
- Enterprise: Custom pricing, unlimited calls, dedicated infrastructure, SLA

CloudMind supports Python, JavaScript, and Java SDKs. The platform includes
pre-trained models for NLP, computer vision, and speech recognition. All tiers
include automatic scaling and 99.9% uptime guarantee.
"""
    
    questions = [
        "What programming languages does CloudMind support?",
        "How much does the Professional tier cost?",
        "Does CloudMind offer a free trial?",  # Not in context
        "What's included in the Enterprise tier?"
    ]
    
    for q in questions:
        result = qa_system.answer_with_confidence(q, context, "tech products")
        print(f"\nQ: {q}")
        print(f"A: {result['answer']}")
        print(f"Confidence: {result['confidence']}")
    
    # Example 2: Technical documentation Q&A
    print("\n\n2. TECHNICAL DOCUMENTATION Q&A")
    print("-" * 50)
    
    tech_context = """
The Transformer architecture consists of an encoder and decoder. The encoder 
processes the input sequence and creates representations. The decoder generates 
the output sequence. Both use self-attention mechanisms and feed-forward networks.

Key components:
- Multi-head attention: Allows the model to focus on different positions
- Positional encoding: Adds position information to embeddings
- Layer normalization: Stabilizes training
- Residual connections: Help with gradient flow

The model uses 6 encoder and 6 decoder layers by default.
"""
    
    tech_questions = [
        "What are the main components of a Transformer?",
        "How many encoder layers does a standard Transformer have?",
        "What is the purpose of positional encoding?",
        "Does the Transformer use LSTM cells?"  # Testing negative case
    ]
    
    for q in tech_questions:
        result = qa_system.answer_with_confidence(q, tech_context, "machine learning")
        print(f"\nQ: {q}")
        print(f"A: {result['answer']}")
        print(f"Confidence: {result['confidence']}")
    
    # Example 3: Simple Q&A without context
    print("\n\n3. ZERO-SHOT QUESTION ANSWERING")
    print("-" * 50)
    
    general_questions = [
        "What is the capital of France?",
        "How do plants produce energy?",
        "What is 15% of 200?"
    ]
    
    for q in general_questions:
        # For zero-shot, we'll use a simpler approach
        prompt = f"Question: {q}\nAnswer:"
        response = qa_system.model(
            prompt,
            max_new_tokens=50,
            temperature=0.5,
            do_sample=True,
            pad_token_id=qa_system.model.tokenizer.eos_token_id
        )
        answer = response[0]['generated_text'].split("Answer:")[-1].strip()
        print(f"\nQ: {q}")
        print(f"A: {answer}")
    
    print("\n" + "=" * 50)
    print("Question answering examples completed!")


if __name__ == "__main__":
    run_question_answering_examples()