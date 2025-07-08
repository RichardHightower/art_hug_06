"""Multi-stage document processing pipeline."""

from transformers import pipeline
from typing import Dict, Any
import json
from config import DEVICE


class DocumentProcessor:
    """Multi-stage document processing pipeline."""
    
    def __init__(self, model=None):
        """Initialize the document processor."""
        if model is None:
            self.model = pipeline(
                "text-generation",
                model="gpt2",
                device=0 if DEVICE == "cuda" else -1
            )
        else:
            self.model = model
            
    def process_document(self, document: str, output_format: str = "report") -> Dict[str, Any]:
        """Process document through multiple stages."""
        
        # Stage 1: Extract key information
        extraction_prompt = f"""Extract the following from this document:
- Main topic
- Key points (up to 5)
- Important dates/deadlines
- Action items

Document: {document}

Format as JSON:"""
        
        # Truncate prompt if too long
        if len(extraction_prompt) > 800:
            extraction_prompt = extraction_prompt[:800] + "..."
            
        extracted = self.model(
            extraction_prompt,
            max_new_tokens=100,
            temperature=0.5,
            do_sample=True,
            pad_token_id=self.model.tokenizer.eos_token_id,
            truncation=True
        )
        extracted_text = extracted[0]['generated_text']
        
        # Stage 2: Analyze sentiment and tone
        sentiment_prompt = f"""Analyze the tone and sentiment of this document:
{document}

Provide:
- Overall sentiment (positive/negative/neutral)
- Tone (formal/casual/urgent/informative)
- Key emotional indicators"""
        
        # Truncate document for sentiment analysis
        if len(document) > 500:
            sentiment_prompt = f"""Analyze the tone and sentiment of this document:
{document[:500]}...

Provide:
- Overall sentiment (positive/negative/neutral)
- Tone (formal/casual/urgent/informative)
- Key emotional indicators"""
            
        sentiment = self.model(
            sentiment_prompt,
            max_new_tokens=80,
            temperature=0.5,
            do_sample=True,
            pad_token_id=self.model.tokenizer.eos_token_id,
            truncation=True
        )
        sentiment_text = sentiment[0]['generated_text']
        
        # Stage 3: Generate formatted output
        if output_format == "report":
            format_prompt = f"""Based on this analysis, create a professional report:

Extracted Information:
{extracted_text}

Sentiment Analysis:
{sentiment_text}

Create a well-structured executive report with:
1. Executive Summary
2. Key Findings
3. Recommendations
4. Next Steps"""
        
        elif output_format == "email":
            format_prompt = f"""Convert this analysis into a professional email:

Information: {extracted_text}

Write a concise email that:
- Summarizes the main points
- Highlights action items
- Maintains appropriate tone
- Includes a clear call-to-action"""
        
        else:  # Default to summary
            format_prompt = f"""Create a concise summary based on:

Extracted Information:
{extracted_text}

Sentiment Analysis:
{sentiment_text}

Provide a clear, actionable summary."""
        
        # Ensure format prompt isn't too long
        if len(format_prompt) > 900:
            # Truncate the extracted and sentiment text if needed
            format_prompt = format_prompt[:900] + "..."
            
        final_output = self.model(
            format_prompt,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.model.tokenizer.eos_token_id,
            truncation=True
        )
        
        return {
            "extracted_info": extracted_text,
            "sentiment": sentiment_text,
            "formatted_output": final_output[0]['generated_text']
        }
    
    def extract_entities(self, document: str) -> Dict[str, Any]:
        """Extract named entities from document."""
        entity_prompt = f"""Extract the following entities from this document:
- People mentioned
- Organizations
- Locations
- Dates
- Monetary values

Document: {document}

List each category:"""
        
        response = self.model(
            entity_prompt,
            max_new_tokens=150,
            temperature=0.3,
            do_sample=True,
            pad_token_id=self.model.tokenizer.eos_token_id
        )
        
        return {"entities": response[0]['generated_text']}
    
    def summarize_by_section(self, document: str) -> Dict[str, Any]:
        """Summarize document section by section."""
        section_prompt = f"""Break down this document into logical sections and summarize each:

Document: {document}

Section summaries:"""
        
        response = self.model(
            section_prompt,
            max_new_tokens=250,
            temperature=0.5,
            do_sample=True,
            pad_token_id=self.model.tokenizer.eos_token_id
        )
        
        return {"section_summaries": response[0]['generated_text']}


def demo_document_processing():
    """Demonstrate document processing capabilities."""
    print("Document Processing Pipeline Demo")
    print("=" * 50)
    
    processor = DocumentProcessor()
    
    # Sample documents
    documents = {
        "business_update": """
Team,

Following our Q3 review, I wanted to share some critical updates. Our revenue 
exceeded targets by 15%, reaching $4.2M. However, customer churn increased to 
8%, primarily due to onboarding issues.

Immediate action required:
1. Review and revamp onboarding process by Nov 15
2. Schedule customer feedback sessions next week
3. Prepare retention strategy presentation for board meeting on Nov 20

The competitive landscape is intensifying, but our product differentiation 
remains strong. We must act quickly to maintain our market position.

Best regards,
Sarah Chen
VP of Product
""",
        
        "technical_report": """
System Performance Analysis - October 2024

Executive Summary:
Our infrastructure has shown 99.8% uptime this month, exceeding our SLA 
requirements. However, response times have degraded by 12% due to increased 
traffic.

Key Findings:
- Database queries are the primary bottleneck
- CDN cache hit rate is only 72% (target: 85%)
- API response times average 250ms (target: 200ms)

Recommendations:
1. Implement database query optimization
2. Review and update CDN caching rules
3. Consider horizontal scaling for API servers

Timeline: Complete optimizations by end of Q4 2024.
""",
        
        "customer_feedback": """
Product Review Summary - Mobile App v3.2

We've analyzed 500+ customer reviews from the past month. Overall satisfaction 
has improved to 4.2/5 stars, up from 3.8 in the previous version.

Positive feedback focuses on:
- Improved UI design (mentioned by 78% of positive reviews)
- Faster load times (65% mentions)
- New features like dark mode (82% approval)

Areas for improvement:
- Battery consumption still high (45% of complaints)
- Sync issues with desktop version (30% of complaints)
- Limited offline functionality (25% requests)

Suggested priorities for v3.3:
1. Optimize battery usage
2. Fix sync reliability
3. Expand offline capabilities
"""
    }
    
    # Example 1: Process business update as report
    print("\n1. BUSINESS UPDATE → EXECUTIVE REPORT")
    print("-" * 50)
    
    result = processor.process_document(documents["business_update"], output_format="report")
    print("Formatted Output:")
    print(result["formatted_output"])
    
    # Example 2: Process technical report as email
    print("\n\n2. TECHNICAL REPORT → EMAIL")
    print("-" * 50)
    
    result = processor.process_document(documents["technical_report"], output_format="email")
    print("Email Output:")
    print(result["formatted_output"])
    
    # Example 3: Extract entities
    print("\n\n3. ENTITY EXTRACTION")
    print("-" * 50)
    
    entities = processor.extract_entities(documents["business_update"])
    print("Extracted Entities:")
    print(entities["entities"])
    
    # Example 4: Section-by-section summary
    print("\n\n4. SECTION-BY-SECTION SUMMARY")
    print("-" * 50)
    
    sections = processor.summarize_by_section(documents["customer_feedback"])
    print("Section Summaries:")
    print(sections["section_summaries"])
    
    # Example 5: Multi-document processing
    print("\n\n5. MULTI-DOCUMENT BATCH PROCESSING")
    print("-" * 50)
    
    print("Processing all documents as summaries...")
    for doc_name, doc_content in documents.items():
        print(f"\n{doc_name.upper()}:")
        result = processor.process_document(doc_content, output_format="summary")
        # Show just the final output
        output = result["formatted_output"]
        if "Provide a clear, actionable summary." in output:
            summary = output.split("Provide a clear, actionable summary.")[-1].strip()
        else:
            summary = output[len(doc_content):].strip()
        print(summary[:200] + "..." if len(summary) > 200 else summary)
    
    print("\n" + "=" * 50)
    print("Document processing demo completed!")


if __name__ == "__main__":
    demo_document_processing()