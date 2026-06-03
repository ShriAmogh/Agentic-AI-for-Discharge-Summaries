import json
import nltk
from app.models import DischargeSummary

# Ensure nltk edit distance works (no extra download needed for standard edit_distance)
from nltk.metrics.distance import edit_distance

def calculate_schema_edit_distance(draft: DischargeSummary, reviewed: DischargeSummary) -> int:
    """Calculates the Levenshtein edit distance between the JSON representations of two DischargeSummaries."""
    draft_json = draft.model_dump_json(indent=2)
    reviewed_json = reviewed.model_dump_json(indent=2)
    
    return edit_distance(draft_json, reviewed_json)

def calculate_flag_recall(draft: DischargeSummary, reviewed: DischargeSummary) -> float:
    """Calculates the recall for escalation flags. 
    Recall = (Draft Flags that are also in Reviewed Flags) / (Total Reviewed Flags)
    If Reviewed has 0 flags, recall is 1.0.
    """
    reviewed_flags = set(reviewed.escalations)
    if not reviewed_flags:
        return 1.0
        
    draft_flags = set(draft.escalations)
    true_positives = len(draft_flags.intersection(reviewed_flags))
    
    return true_positives / len(reviewed_flags)
