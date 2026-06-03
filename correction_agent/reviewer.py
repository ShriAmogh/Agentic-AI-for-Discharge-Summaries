import json
import os
from google import genai
from pydantic import BaseModel, Field
from typing import List
from app.models import DischargeSummary
from app.config import REVIEWER_MODEL

GEMINI_API_KEY = os.getenv("GEMINI_KEY")

class ReviewerOutput(BaseModel):
    corrected_summary: DischargeSummary
    correction_reasons: List[str] = Field(description="A list of explicit reasons explaining what the original draft got wrong. If it was perfect, return an empty list.")

class SimulatedReviewer:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("GEMINI_KEY is missing.")
        self.client = genai.Client(api_key=self.api_key)
    
    def review_draft(self, draft: DischargeSummary, source_text: str) -> tuple[DischargeSummary, List[str]]:
        prompt = f"""You are a strict attending physician reviewing a Medical AI's draft Discharge Summary.
You must compare the Draft against the original Source Text to ensure absolute accuracy and adherence to safety guardrails.

**REVIEWER POLICY**:
1. Check for Fabrications: Are there any facts in the draft that do NOT exist in the source text?
2. Check for Missing Information: Did the agent miss any labs, diagnoses, or undocumented medication changes?
3. Check for Escaltion Failures: Did the agent fail to escalate undocumented medication changes or conflicts?

Source Text:
{source_text}

Agent Draft (JSON):
{draft.model_dump_json(indent=2)}

Please fix any errors in the draft to create a perfect, safe 'corrected_summary'.
Also, generate a list of 'correction_reasons'. For each mistake found in the draft, write a short, clear instruction on how to avoid it (e.g., 'Do not fabricate a reason for starting carvedilol if it is not documented in the notes.').
"""
        
        try:
            response = self.client.models.generate_content(
                model=REVIEWER_MODEL,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ReviewerOutput,
                )
            )
            
            output_data = json.loads(response.text)
            output = ReviewerOutput(**output_data)
            return output.corrected_summary, output.correction_reasons
        except Exception as e:
            print(f"Reviewer encountered an error: {e}")
            return draft, []
