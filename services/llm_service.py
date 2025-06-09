from litellm import completion
from fastapi import HTTPException
import re
from datetime import datetime

from config import config
from models import UserDetails


class LLMService:
    def __init__(self):
        self.model = config.LLM_MODEL

    def format_user_profile(self, user_details: UserDetails) -> str:
        """Format user profile with available information, handling missing details gracefully"""
        profile_parts = []

        # Handle age
        if user_details.age is not None:
            profile_parts.append(f"{user_details.age}-year-old")
        else:
            profile_parts.append("Patient of unspecified age")

        # Handle gender
        if user_details.gender:
            profile_parts.append(user_details.gender.strip())
        else:
            profile_parts.append("of unspecified gender")

        # Handle blood type
        if user_details.blood_type:
            profile_parts.append(f"Blood type {user_details.blood_type.strip()}")
        else:
            profile_parts.append("Blood type not provided")

        # Handle bio/medical background
        if user_details.bio and user_details.bio.strip():
            medical_background = f"Medical background: {user_details.bio.strip()}"
        else:
            medical_background = "Medical background: No additional medical history provided"

        # Combine profile parts
        basic_profile = " ".join(profile_parts)
        return f"{basic_profile}. {medical_background}"

    def check_missing_details(self, user_details: UserDetails) -> list:
        """Check which user details are missing and return a list of missing fields"""
        missing_fields = []

        if not user_details.age:
            missing_fields.append("age")
        if not user_details.gender or not user_details.gender.strip():
            missing_fields.append("gender")
        if not user_details.blood_type or not user_details.blood_type.strip():
            missing_fields.append("blood_type")
        if not user_details.bio or not user_details.bio.strip():
            missing_fields.append("medical_history")

        return missing_fields

    def create_medical_prompt(self, user_details: UserDetails, condition: str,
                              probability: float, weaviate_content: str) -> str:
        """Create a structured medical consultation prompt, handling missing user details"""

        # Format patient profile with available information
        patient_profile = self.format_user_profile(user_details)

        # Check for missing details
        missing_details = self.check_missing_details(user_details)

        # Create additional instruction for missing details
        missing_details_instruction = ""
        if missing_details:
            missing_list = ", ".join(missing_details)
            missing_details_instruction = f"""
Note: Some patient details are missing ({missing_list}). Please acknowledge these missing details in your assessment and recommend that the patient provides this information for a more comprehensive evaluation."""

        prompt = f"""You are a doctor providing a medical consultation. Write a formal report following this exact structure:

Re: Condition Consultation: {condition}

Dear [Patient's Name],

Thank you for choosing our medical facility for your healthcare needs. Based on the diagnostic findings and medical history you have provided, it appears that you have a condition known as {condition}.

Patient Profile: {patient_profile}

Our diagnosis is supported by the confidence score ({probability:.3f}) assigned by our clinical decision support tool.

Clinical Reference: {weaviate_content}

{missing_details_instruction}

Please provide the following sections:

1. Condition Overview:
   a) Pathophysiology and clinical presentation
   b) Typical demographic and risk factors  
   c) Differential diagnosis considerations

2. Clinical Assessment:
   a) Key diagnostic features and criteria
   b) Expected symptom progression
   c) Potential complications if untreated

3. Recommendations:
   a) Immediate steps and follow-up care
   b) Lifestyle modifications if applicable
   c) When to seek urgent medical attention

Keep each subsection concise (2-3 sentences). If condition is serious, recommend immediate medical consultation. If patient details are incomplete, mention the importance of providing complete medical information for accurate diagnosis and treatment planning."""

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
