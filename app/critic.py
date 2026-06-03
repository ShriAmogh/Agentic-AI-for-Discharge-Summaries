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
        prompt = f"""You are a strict Medical Safety Critic. Your job is to review the generated Discharge Summary draft against the raw Source Text to catch critical clinical safety risks before final output generation.

### 1. SAFETY EVALUATION CHECKLIST
1. **Unescalated Medication Changes:** Are there any undocumented changes, dose adjustments, or early cessations in the source text that were not escalated or flagged?
2. **Ignored Diagnostic Conflicts:** Are there severe diagnostic contradictions (e.g., DKA in ER notes vs. Gastroenteritis in Ward notes) that the agent completely ignored or left out of the Active Diagnoses/Conflicts sections?
3. **Incomplete Text/Truncated Strings:** Look for raw, truncated medication strings containing unclosed punctuation (e.g., 'TAB. ENTR(' or 'CAP. SPIRIV['). 
   * *CRITICAL COMPLIANCE:* If the agent included the truncated string but appended a verification flag (e.g., '[INCOMPLETE STRING / PHARMACY VERIFICATION REQUIRED]'), you MUST treat it as SAFE.

### 2. CRITICAL BYPASS RULE (Missing Home Medications)
If the raw Source Text completely lacks a baseline home medication list, the agent is structurally trapped. 
You **MUST** mark the lack of baseline medications as **SAFE** (and NOT intervene for this reason) if the agent has explicitly included this exact structural warning anywhere in the draft:
"Baseline Home Medications: NOT AVAILABLE IN SOURCE DOCUMENTATION. Note: Pre-admission baseline could not be verified; discharge medications listed above represent the complete active regimen moving forward."

### 3. CONTEXT FOR REVIEW
**Source Text:**
{source_text}

**Draft (JSON):**
{draft.model_dump_json(indent=2)}

### 4. OUTPUT INSTRUCTIONS
* If you find a severe clinical safety risk that has *not* been accounted for or flagged by the agent via the rules above, you must:
  1. Set `is_safe` to `False`.
  2. Provide a specific, actionable `critique_reason`.
  3. Output a `corrected_draft` JSON where these issues are explicitly added to the escalations, conflicts, or notes section.
* If the draft respects the bypass rules, acknowledges the diagnostic conflicts, and tags truncated string fragments appropriately, set `is_safe` to `True` and return the draft as-is.
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
