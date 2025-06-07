from litellm import completion
from fastapi import HTTPException
import re
from datetime import datetime

from config import config
from models import UserDetails


class LLMService:
    def __init__(self):
        self.model = config.LLM_MODEL

    def create_medical_prompt(self, user_details: UserDetails, condition: str,
                              probability: float, weaviate_content: str) -> str:
        """Create a structured medical consultation prompt"""
        prompt = f"""You are a doctor providing a medical consultation. Write a formal report following this exact structure:

Re: Condition Consultation: {condition}

Dear [Patient's Name],

Thank you for choosing our medical facility for your healthcare needs. Based on the diagnostic findings and medical history you have provided, it appears that you have a condition known as {condition}.

Patient Profile: {user_details.age}-year-old {user_details.gender}, Blood type {user_details.blood_type}. Medical background: {user_details.bio}

Our diagnosis is supported by the confidence score ({probability:.3f}) assigned by our clinical decision support tool.

Clinical Reference: {weaviate_content}

Please provide the following sections:

1. Condition Overview:
   a) Pathophysiology and clinical presentation
   b) Typical demographic and risk factors  
   c) Differential diagnosis considerations

2. Clinical Assessment:
   a) Key diagnostic features and criteria
   b) Expected symptom progression
   c) Potential complications if untreated

Keep each subsection concise (2-3 sentences). If condition is serious, recommend immediate medical consultation."""

        return prompt

    def convert_to_html(self, text: str) -> str:
        """Convert plain text response to HTML if needed"""
        # If already contains HTML tags, return as-is
        if '<p>' in text or '<div>' in text or '<ul>' in text:
            return text

        # Basic text to HTML conversion
        html = text

        # Convert headings (lines that end with :)
        html = re.sub(r'^([A-Z][A-Z\s]+):$', r'<h3>\1</h3>', html, flags=re.MULTILINE)

        # Convert bullet points
        html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)

        # Wrap consecutive <li> tags in <ul>
        html = re.sub(r'(<li>.*?</li>)(\s*<li>.*?</li>)*', r'<ul>\g<0></ul>', html, flags=re.DOTALL)

        # Convert double line breaks to paragraph breaks
        html = re.sub(r'\n\n+', '</p><p>', html)

        # Wrap in paragraphs
        html = f'<p>{html}</p>'

        # Clean up empty paragraphs
        html = re.sub(r'<p>\s*</p>', '', html)

        # Convert **bold** to <strong>
        html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)

        # Convert *italic* to <em>
        html = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', html)

        return html

    def add_css_classes(self, html: str) -> str:
        """Add CSS classes to HTML elements for consistent styling"""
        # Add classes to common elements
        html = html.replace('<p>', '<p class="my-paragraph" dir="ltr">')
        html = html.replace('<ul>', '<ul class="list-disc list-inside">')
        html = html.replace('<strong>', '<strong class="font-bold">')
        html = html.replace('<h2>', '<h2 class="text-xl font-bold mb-4">')
        html = html.replace('<h3>', '<h3 class="text-lg font-semibold mb-2">')

        # Add white-space preservation to spans
        html = re.sub(r'(<li[^>]*>)(.*?)(</li>)',
                      r'\1<span style="white-space: pre-wrap;">\2</span>\3', html)
        html = re.sub(r'(<p[^>]*>)(.*?)(</p>)',
                      r'\1<span style="white-space: pre-wrap;">\2</span>\3', html)

        return html

    def generate_response(self, prompt: str) -> str:
        """Call the LLM with the constructed prompt and return HTML formatted response"""
        try:
            response = completion(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )

            raw_response = response.choices[0].message.content

            # Convert to HTML if not already formatted
            html_response = self.convert_to_html(raw_response)

            # Add CSS classes for styling
            formatted_response = self.add_css_classes(html_response)

            return formatted_response

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calling LLM: {str(e)}")
