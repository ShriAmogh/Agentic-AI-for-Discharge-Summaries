import os
from dotenv import load_dotenv

# Ensure environment variables are loaded from .env
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_KEY")

# Model Configurations
OCR_MODEL = "models/gemini-3.5-flash"
DRUG_DETECTION_MODEL = "models/gemini-flash-lite-latest"
AGENT_MODEL = "models/gemini-flash-lite-latest"
REVIEWER_MODEL = "models/gemini-flash-lite-latest"
SYNTHETIC_DATA_MODEL = "models/gemini-flash-lite-latest"
CRITIC_MODEL = "models/gemini-flash-lite-latest"

# Agent Configurations
MAX_STEPS = 8
MAX_PAGES = 12

# Learning Loop / Correction Agent Configurations
SYNTHETIC_DATA_PATHS = [
    "docs/synthetic_data/synthetic_patient_1.json",
    "docs/synthetic_data/synthetic_patient_2.json",
    "docs/synthetic_data/synthetic_patient_3.json"
]
TEST_PDF_PATH = "docs/patient_report.pdf"
MEMORY_FILE_PATH = "docs/memory/correction_memory.json"
TRAIN_TOP_K_MISTAKES = 15
TEST_TOP_K_MISTAKES = 15
