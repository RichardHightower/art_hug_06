"""Conversational AI examples with specialized assistants."""

from transformers import pipeline
from typing import List
from config import DEVICE


class ConversationalAssistant:
    """Domain-specific conversational agent with role prompting and memory."""
    
    def __init__(self, model=None, role: str = "", personality: str = ""):
        """Initialize the conversational assistant."""
        if model is None:
            self.model = pipeline(
                "text-generation",
                model="gpt2",  # Using GPT-2 for better compatibility
                device=0 if DEVICE == "cuda" else -1
            )
        else:
            self.model = model
            
        self.role = role
        self.personality = personality
        self.conversation_history: List[str] = []
        self.max_history = 5  # Keep last 5 exchanges
        
    def get_system_prompt(self) -> str:
        """Get the system prompt for this assistant."""
        return f"""You are {self.role}. {self.personality}

Guidelines:
- Stay in character
- Be helpful but maintain appropriate boundaries
- Use domain-specific terminology when relevant
- Keep responses concise but informative

Current conversation:"""
    
    def chat(self, user_input: str) -> str:
        """Process user input and generate response."""
        # Add user input to history
        self.conversation_history.append(f"User: {user_input}")
        
        # Construct full prompt with history
        full_prompt = self.get_system_prompt() + "\n"
        
        # Include recent history
        start_idx = max(0, len(self.conversation_history) - self.max_history * 2)
        for msg in self.conversation_history[start_idx:]:
            full_prompt += msg + "\n"
        
        full_prompt += "Assistant:"
        
        # Limit prompt length to avoid model limits
        if len(full_prompt) > 800:
            # Keep only recent history
            full_prompt = self.get_system_prompt() + "\n"
            start_idx = max(0, len(self.conversation_history) - 2)
            for msg in self.conversation_history[start_idx:]:
                full_prompt += msg + "\n"
            full_prompt += "Assistant:"
        
        # Generate response
        response = self.model(
            full_prompt,
            max_new_tokens=80,
            temperature=0.8,
            do_sample=True,
            pad_token_id=self.model.tokenizer.eos_token_id,
            truncation=True
        )
        
        # Extract only the new response
        full_response = response[0]['generated_text']
        if "Assistant:" in full_response:
            assistant_response = full_response.split("Assistant:")[-1].strip()
        else:
            assistant_response = full_response[len(full_prompt):].strip()
        
        # Add to history
        self.conversation_history.append(f"Assistant: {assistant_response}")
        
        return assistant_response
    
    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []


def run_conversational_ai_examples():
    """Run conversational AI examples with different specialized assistants."""
    print("Initializing Conversational AI Examples...")
    
    # Create specialized assistants
    assistants = {
        "medical": ConversationalAssistant(
            role="a medical information assistant",
            personality="You are knowledgeable, empathetic, and always remind users to consult healthcare professionals for personal medical advice"
        ),
        "tech_support": ConversationalAssistant(
            role="a technical support specialist",
            personality="You are patient, detail-oriented, and skilled at explaining technical concepts in simple terms"
        ),
        "tutor": ConversationalAssistant(
            role="a friendly math tutor",
            personality="You are encouraging, break down problems step-by-step, and use examples to explain concepts"
        ),
        "chef": ConversationalAssistant(
            role="a professional chef",
            personality="You are creative, passionate about food, and enjoy sharing cooking tips and recipes"
        )
    }
    
    # Example 1: Medical Assistant
    print("\n1. MEDICAL ASSISTANT DEMO")
    print("-" * 50)
    
    medical_conversations = [
        "I've been having headaches lately",
        "What might cause them?",
        "Should I be worried?"
    ]
    
    medical_assistant = assistants["medical"]
    for user_input in medical_conversations:
        print(f"\nUser: {user_input}")
        response = medical_assistant.chat(user_input)
        print(f"Assistant: {response}")
    
    # Example 2: Tech Support
    print("\n\n2. TECH SUPPORT DEMO")
    print("-" * 50)
    
    tech_conversations = [
        "My computer is running slowly",
        "I haven't restarted in weeks",
        "How do I check what's using memory?"
    ]
    
    tech_support = assistants["tech_support"]
    for user_input in tech_conversations:
        print(f"\nUser: {user_input}")
        response = tech_support.chat(user_input)
        print(f"Assistant: {response}")
    
    # Example 3: Math Tutor
    print("\n\n3. MATH TUTOR DEMO")
    print("-" * 50)
    
    tutor_conversations = [
        "Can you help me understand fractions?",
        "What's 1/2 + 1/3?",
        "Why do we need a common denominator?"
    ]
    
    tutor = assistants["tutor"]
    for user_input in tutor_conversations:
        print(f"\nUser: {user_input}")
        response = tutor.chat(user_input)
        print(f"Assistant: {response}")
    
    # Example 4: Context-aware conversation
    print("\n\n4. CONTEXT-AWARE CONVERSATION (CHEF)")
    print("-" * 50)
    
    chef_conversations = [
        "I want to make pasta for dinner",
        "I have tomatoes, garlic, and basil",
        "How long should I cook it?",
        "Any tips for making it restaurant-quality?"
    ]
    
    chef = assistants["chef"]
    for user_input in chef_conversations:
        print(f"\nUser: {user_input}")
        response = chef.chat(user_input)
        print(f"Assistant: {response}")
    
    # Example 5: Conversation reset demonstration
    print("\n\n5. CONVERSATION RESET DEMO")
    print("-" * 50)
    
    print("Starting new conversation with tech support...")
    tech_support.reset_conversation()
    
    new_conversation = [
        "Hi, I need help with my printer",
        "It's not printing anything",
        "The lights are on but nothing happens"
    ]
    
    for user_input in new_conversation:
        print(f"\nUser: {user_input}")
        response = tech_support.chat(user_input)
        print(f"Assistant: {response}")
    
    print("\n" + "=" * 50)
    print("Conversational AI examples completed!")


if __name__ == "__main__":
    run_conversational_ai_examples()