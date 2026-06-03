import json
from pydantic import BaseModel, Field
from google import genai
from app.config import GEMINI_API_KEY, CRITIC_MODEL
from app.models import DischargeSummary
from app.tools import read_pdf_pages

class CriticOutput(BaseModel):
    is_safe: bool = Field(description="True if the draft is completely safe, False if there are unescalated conflicts or undocumented medication changes.")
    critique_reason: str = Field(description="Explanation of what is wrong or why it is safe.")
    corrected_draft: DischargeSummary = Field(description="A revised draft with escalations added if needed.")

class CriticAgent:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("GEMINI_KEY is missing.")
        self.client = genai.Client(api_key=self.api_key)
        
    def reflect(self, draft: DischargeSummary, pdf_path: str = "", source_text: str = "") -> DischargeSummary:
        """Runs a self-reflection pass on the draft before finalizing it."""
        if not source_text:
            source_text = read_pdf_pages(pdf_path=pdf_path)
        prompt = f"""You are a safety Critic. Review this Discharge Summary draft against its source text.
Look specifically for:
1. Are there any undocumented medication changes that have NOT been escalated?
2. Is there conflicting information from different notes that has NOT been escalated?

Source Text:
{source_text}

Draft:
{draft.model_dump_json(indent=2)}

If you find failures to escalate, set is_safe to False, provide a critique_reason, and output a corrected_draft where those issues are properly added to the `escalations` list.
If it is safe, set is_safe to True and return the draft as-is.
"""
        try:
            response = self.client.models.generate_content(
                model=CRITIC_MODEL,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=CriticOutput,
                )
            )
            
            output = CriticOutput.model_validate_json(response.text)
            if not output.is_safe:
                print(f"⚠️ Critic intervened! Reason: {output.critique_reason}")
            return output.corrected_draft
        except Exception as e:
            print(f"Critic encountered an error: {e}")
            return draft
