"""Production-ready prompt management system."""

import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any


class ProductionPromptManager:
    """Production-ready prompt management system with versioning and analytics."""
    
    def __init__(self, model=None):
        """Initialize the prompt manager."""
        self.model = model
        self.prompt_versions: Dict[str, Dict[str, Any]] = {}
        self.usage_logs: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def register_prompt(
        self, 
        name: str, 
        version: str, 
        template: str, 
        metadata: Optional[Dict] = None
    ):
        """Register a new prompt version."""
        key = f"{name}_v{version}"
        self.prompt_versions[key] = {
            "template": template,
            "metadata": metadata or {},
            "created_at": datetime.now(),
            "usage_count": 0,
            "avg_latency": 0,
            "success_rate": 1.0
        }
        self.logger.info(f"Registered prompt: {key}")
        
    def execute_prompt(
        self, 
        name: str, 
        version: str, 
        variables: Dict[str, Any], 
        **generation_kwargs
    ) -> Dict[str, Any]:
        """Execute a prompt with monitoring."""
        key = f"{name}_v{version}"
        
        if key not in self.prompt_versions:
            raise ValueError(f"Prompt {key} not found")
        
        start_time = time.time()
        prompt_data = self.prompt_versions[key]
        
        try:
            # Format prompt with variables
            prompt = prompt_data["template"].format(**variables)
            
            # Generate response (mock if no model provided)
            if self.model:
                response = self.model(prompt, **generation_kwargs)
                response_text = response[0]['generated_text']
            else:
                # Mock response for demonstration
                response_text = f"Mock response for prompt: {name} v{version}"
            
            # Calculate metrics
            latency = time.time() - start_time
            success = True
            
            # Update metrics
            prompt_data["usage_count"] += 1
            prompt_data["avg_latency"] = (
                (prompt_data["avg_latency"] * (prompt_data["usage_count"] - 1) + latency) 
                / prompt_data["usage_count"]
            )
            
            # Log usage
            self.usage_logs.append({
                "prompt_key": key,
                "timestamp": datetime.now(),
                "latency": latency,
                "success": success,
                "input_length": len(prompt),
                "output_length": len(response_text)
            })
            
            return {
                "response": response_text,
                "metrics": {
                    "latency": latency,
                    "prompt_version": key,
                    "timestamp": datetime.now()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error executing prompt {key}: {str(e)}")
            prompt_data["success_rate"] *= 0.95  # Decay success rate
            raise
            
    def get_best_prompt(self, name: str) -> Optional[str]:
        """Get best performing prompt version."""
        versions = [k for k in self.prompt_versions.keys() if k.startswith(name)]
        
        if not versions:
            return None
            
        # Score based on success rate and latency
        best_version = max(versions, key=lambda v: 
            self.prompt_versions[v]["success_rate"] / 
            (self.prompt_versions[v]["avg_latency"] + 1)
        )
        
        return best_version
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get prompt performance analytics."""
        return {
            "total_prompts": len(self.prompt_versions),
            "total_executions": len(self.usage_logs),
            "prompt_performance": {
                k: {
                    "usage_count": v["usage_count"],
                    "avg_latency": round(v["avg_latency"], 3),
                    "success_rate": round(v["success_rate"], 3)
                }
                for k, v in self.prompt_versions.items()
            }
        }
    
    def get_prompt_history(self, name: str) -> List[Dict[str, Any]]:
        """Get execution history for a specific prompt."""
        history = []
        for log in self.usage_logs:
            if log["prompt_key"].startswith(name):
                history.append(log)
        return history
    
    def compare_versions(self, name: str) -> Dict[str, Any]:
        """Compare all versions of a prompt."""
        versions = [k for k in self.prompt_versions.keys() if k.startswith(name)]
        
        comparison = {}
        for version in versions:
            data = self.prompt_versions[version]
            comparison[version] = {
                "usage_count": data["usage_count"],
                "avg_latency": round(data["avg_latency"], 3),
                "success_rate": round(data["success_rate"], 3),
                "created_at": data["created_at"].strftime("%Y-%m-%d %H:%M:%S")
            }
        
        return comparison


def demo_prompt_manager():
    """Demonstrate prompt management capabilities."""
    print("Production Prompt Management Demo")
    print("=" * 50)
    
    # Initialize manager
    pm = ProductionPromptManager()
    
    # Register multiple prompt versions
    print("\n1. REGISTERING PROMPT VERSIONS")
    print("-" * 50)
    
    pm.register_prompt(
        "customer_email",
        "1.0",
        "Write a professional email response to: {complaint}\nTone: {tone}",
        {"author": "team_a", "tested": True}
    )
    
    pm.register_prompt(
        "customer_email",
        "2.0",
        """You are a customer service representative. 
Respond professionally to this complaint: {complaint}
Use a {tone} tone and include next steps.""",
        {"author": "team_b", "tested": True}
    )
    
    pm.register_prompt(
        "customer_email",
        "2.1",
        """You are an experienced customer service representative. 
Customer complaint: {complaint}

Please respond with:
1. Acknowledgment of their concern
2. A {tone} response
3. Clear next steps
4. Contact information for follow-up""",
        {"author": "team_b", "tested": True, "improved": True}
    )
    
    print("Registered 3 versions of 'customer_email' prompt")
    
    # Execute prompts
    print("\n2. EXECUTING PROMPTS")
    print("-" * 50)
    
    complaint = "My order hasn't arrived after 2 weeks"
    
    for version in ["1.0", "2.0", "2.1"]:
        result = pm.execute_prompt(
            "customer_email",
            version,
            {"complaint": complaint, "tone": "empathetic"},
            max_new_tokens=150
        )
        print(f"\nVersion {version}:")
        print(f"Response: {result['response']}")
        print(f"Latency: {result['metrics']['latency']:.3f}s")
    
    # Simulate more usage for analytics
    print("\n3. SIMULATING PRODUCTION USAGE")
    print("-" * 50)
    
    complaints = [
        "Product arrived damaged",
        "Wrong item received",
        "Refund not processed",
        "Account access issues"
    ]
    
    import random
    for _ in range(10):
        version = random.choice(["1.0", "2.0", "2.1"])
        complaint = random.choice(complaints)
        try:
            pm.execute_prompt(
                "customer_email",
                version,
                {"complaint": complaint, "tone": "professional"}
            )
        except:
            pass  # Simulate some failures
    
    # Get analytics
    print("\n4. ANALYTICS REPORT")
    print("-" * 50)
    
    analytics = pm.get_analytics()
    print(f"Total prompts registered: {analytics['total_prompts']}")
    print(f"Total executions: {analytics['total_executions']}")
    print("\nPerformance by version:")
    for version, metrics in analytics['prompt_performance'].items():
        print(f"\n{version}:")
        print(f"  - Usage count: {metrics['usage_count']}")
        print(f"  - Avg latency: {metrics['avg_latency']}s")
        print(f"  - Success rate: {metrics['success_rate']}")
    
    # Get best performing version
    best = pm.get_best_prompt("customer_email")
    print(f"\nBest performing version: {best}")
    
    # Compare versions
    print("\n5. VERSION COMPARISON")
    print("-" * 50)
    
    comparison = pm.compare_versions("customer_email")
    for version, data in comparison.items():
        print(f"\n{version}:")
        for key, value in data.items():
            print(f"  - {key}: {value}")
    
    # Additional prompt examples
    print("\n6. ADDITIONAL PROMPT TYPES")
    print("-" * 50)
    
    # Register different prompt types
    pm.register_prompt(
        "product_description",
        "1.0",
        "Write a compelling product description for: {product}\nKey features: {features}",
        {"type": "marketing"}
    )
    
    pm.register_prompt(
        "code_review",
        "1.0",
        "Review this code and provide feedback:\n{code}\nFocus on: {focus_areas}",
        {"type": "technical"}
    )
    
    pm.register_prompt(
        "meeting_summary",
        "1.0",
        "Summarize this meeting transcript:\n{transcript}\nHighlight: {key_points}",
        {"type": "business"}
    )
    
    print("Registered additional prompt types: product_description, code_review, meeting_summary")
    
    print("\n" + "=" * 50)
    print("Prompt management demo completed!")


if __name__ == "__main__":
    demo_prompt_manager()