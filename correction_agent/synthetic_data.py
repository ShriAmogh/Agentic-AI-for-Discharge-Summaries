import os
import json
from google import genai
from app.config import GEMINI_API_KEY, SYNTHETIC_DATA_MODEL, SYNTHETIC_DATA_PATHS

def generate_synthetic_patient_text(prompt: str) -> str:
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=SYNTHETIC_DATA_MODEL,
        contents=prompt,
    )
    return response.text.strip()

def generate_synthetic_patients():
    prompts = [
        "Generate a highly realistic, complex, and lengthy clinical chart for a patient with severe Heart Failure exacerbation. Structure it as a JSON object with a single key 'text' containing the full string. The text should mimic multiple pages of hospital records: Admission Note (ER), Daily Progress Notes (Day 1-3), Labs, and a Discharge Plan. Introduce deliberate chaos: an undocumented medication shift (they started carvedilol on day 2 but no reason was given in any note), missing pending labs at discharge, and conflicting weights across notes. Make it at least 800 words of clinical text.",
        
        "Generate a highly realistic, complex, and lengthy clinical chart for a patient with post-operative pneumonia and sepsis. Structure it as a JSON object with a single key 'text' containing the full string. The text should mimic multiple pages: ER triage, ICU transfer note, step-down note, and Discharge Instructions. Introduce deliberate chaos: a conflicting admission date (one note says admitted Jan 2, another says Jan 5), missing discharge condition, and a medication conflict where an antibiotic is stopped without explanation. Make it at least 800 words of clinical text.",
        
        "Generate a highly realistic, complex, and lengthy clinical chart for a patient with Diabetic Ketoacidosis (DKA) and Acute Kidney Injury. Structure it as a JSON object with a single key 'text' containing the full string. The text should mimic multiple pages: Admission H&P, Nephrology Consult, Progress Notes, and Discharge summary draft. Introduce deliberate chaos: a documented severe allergy to Penicillin on page 1, but later in the discharge plan, Amoxicillin is prescribed. Also, leave out the final discharge diagnosis entirely. Make it at least 800 words of clinical text."
    ]
    
    for i, prompt in enumerate(prompts):
        print(f"Generating synthetic patient {i+1}...")
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=SYNTHETIC_DATA_MODEL,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        data_path = SYNTHETIC_DATA_PATHS[i]
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        # Save as JSON
        with open(data_path, 'w') as f:
            f.write(response.text.strip())
            
        print(f"Saved {data_path}")

if __name__ == "__main__":
    generate_synthetic_patients()
