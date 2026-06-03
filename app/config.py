import os
from dotenv import load_dotenv

# Ensure environment variables are loaded from .env
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_KEY")

# Model Configurations
OCR_MODEL = "models/gemini-3.5-flash"
DRUG_DETECTION_MODEL = "models/gemini-flash-lite-latest"
AGENT_MODEL = "models/gemini-flash-lite-latest"

# Agent Configurations
MAX_STEPS = 3
