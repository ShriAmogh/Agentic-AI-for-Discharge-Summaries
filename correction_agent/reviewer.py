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
        prompt = f"""You are a strict, detail-oriented attending physician reviewing a Medical AI's draft Discharge Summary against the raw Source Documentation.
You must enforce absolute clinical accuracy, safety guardrails, and zero-tolerance for hallucinations.

### 1. EVALUATION OBJECTIVES
* **Fabrication Check:** Identify any statements, values, dates, or clinical assertions in the draft that have NO evidence in the Source Text.
* **Omission Check:** Identify missing critical data points (e.g., abnormal labs, secondary diagnoses, or pending tests).
* **Escalation & Reconciliation Check:** Ensure the agent explicitly flagged missing data (like baseline home meds) or conflicting documentation (like cross-page diagnostic discrepancies) instead of smoothing them over.

### 2. SOURCE DATA
**Original Source Text:**
{source_text}

**Agent Draft (JSON):**
{draft.model_dump_json(indent=2)}

### 3. OUTPUT REQUIREMENTS
You must generate:
1. A perfect, safe `corrected_summary` JSON object matching the required schema.
2. A list of `correction_reasons` that the agent can use to tune its weights/prompts.

### 4. CRITICAL: CORRECTION REASONS FORMATTING POLICY
Your `correction_reasons` MUST be high-level structural rules, but they **MUST NOT be lazy platitudes** (e.g., do not just say "Avoid fabrications"). They must pinpoint the *structural logic failure* without leaking case-specific words (names, specific medications like "Amoxicillin", or specific diagnoses like "DKA").

Follow this **Taxonomy Formula** for your reasons:
`[Section Affected] + [Specific Logical/Structural Failure] + [Expected Structural Behavior]`

* **BAD (Too Specific):** "Failed to mention the patient's potassium level of 2.9 in the hospital course."
* **BAD (Too Generic):** "Ensure the hospital course contains accurate clinical data."
* **GOOD (Perfect Meta-Rule):** "In [Hospital Course], the agent omitted critical baseline laboratory values that deviated significantly from normal reference ranges; always extract and track acute metabolic trends sequentially."

* **BAD (Too Specific):** "Ordered Amoxicillin despite a documented Penicillin allergy."
* **BAD (Too Generic):** "Check for safety and medication issues."
* **GOOD (Perfect Meta-Rule):** "In [Discharge Medications], the agent failed to cross-reference the extracted discharge prescription strings against the patient's documented historical allergen contraindications."

* **BAD (Too Specific):** "Copied 'TAB. ENTR(' as a medication name."
* **GOOD (Perfect Meta-Rule):** "In [Discharge Medications], the agent transcribed incomplete string fragments or truncated text characters from the source document without appending a verification or data-corruption flag."

Please output the final corrected summary and the compliant list of meta-rules below:
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
