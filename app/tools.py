from email import contentmanager
import fitz
import os
import json
from google import genai
from .config import GEMINI_API_KEY, OCR_MODEL, DRUG_DETECTION_MODEL
from google.genai import types

def read_pdf_pages(pdf_path: str = "docs/patient_report.pdf", max_pages: int = 4) -> str:
    """Reads the text from the first `max_pages` of the specified PDF using OCR via Gemini Vision."""
    if not os.path.exists(pdf_path):
        return f"Error: File {pdf_path} not found."
    
    if not GEMINI_API_KEY:
        return "Error: GEMINI_KEY is missing. Cannot perform OCR."
        
    try:
        doc = fitz.open(pdf_path)
        content = []
        pages_to_read = min(len(doc), max_pages)
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        for page_num in range(pages_to_read):
            page = doc[page_num]
            # Render page as a high-resolution image
            pix = page.get_pixmap(dpi=200)
            img_data = pix.tobytes("png")
            
            # Use Gemini to perform OCR on the image
            response = client.models.generate_content(
                model=OCR_MODEL,
                contents=[
                    "Extract all the text from this clinical document page exactly as written. Do not summarize or alter the text. Maintain the original structure where possible.",
                    types.Part.from_bytes(
                        data=img_data,
                        mime_type="image/png"
                    )
                ]
)
            text = response.text.strip() if response.text else ""
            content.append(f"--- PAGE {page_num + 1} ---\n{text}")
            
        return "\n\n".join(content)
    except Exception as e:
        return f"Error reading PDF via OCR: {str(e)}"

def drug_interaction_check(medications: str) -> str:
    """Uses LLM to perform a dynamic drug interaction check."""
    if not GEMINI_API_KEY:
        return "Error: GEMINI_KEY is missing. Cannot perform dynamic drug interaction check."
    
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = (
            f"As a clinical pharmacologist, evaluate the following medications for any known "
            f"critical or severe drug interactions: {medications}. "
            "If there are any interactions, list them clearly with severity. "
            "If there are no critical interactions, state 'No known critical interactions found.' "
            "Keep the response concise and strictly about pharmacological interactions."
        )
        response = client.models.generate_content(
            model=DRUG_DETECTION_MODEL,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Error performing drug interaction check: {str(e)}"

def escalate(issue_description: str) -> str:
    """Records an escalation for the clinician."""
    # In a real system, this might write to a database or trigger an alert.
    # Here, we just return a confirmation so the agent knows it succeeded.
    return f"ESCALATED TO CLINICIAN: {issue_description}"

def execute_tool(tool_name: str, action_input: str) -> str:
    """Router for tool execution."""
    if tool_name == "read_pdf_pages":
        return read_pdf_pages()
    elif tool_name == "drug_interaction_check":
        return drug_interaction_check(action_input)
    elif tool_name == "escalate":
        return escalate(action_input)
    else:
        return f"Error: Tool '{tool_name}' not recognized."
