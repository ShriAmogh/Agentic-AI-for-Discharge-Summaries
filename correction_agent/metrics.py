import json
import nltk
from app.models import DischargeSummary

# Ensure nltk edit distance works (no extra download needed for standard edit_distance)
from nltk.metrics.distance import edit_distance

def calculate_schema_edit_distance(draft: DischargeSummary, correct_draft: DischargeSummary) -> int:
    """Calculates the raw Levenshtein distance between two Pydantic schemas."""
    draft_json = draft.model_dump_json()
    correct_json = correct_draft.model_dump_json()
    return edit_distance(draft_json, correct_json)

def calculate_schema_edit_score(draft: DischargeSummary, correct_draft: DischargeSummary) -> float:
    """Calculates a normalized similarity score based on the Levenshtein distance between two Pydantic schemas.
    Score = 1 - (edit_distance / max_len)
    """
    draft_json = draft.model_dump_json()
    correct_json = correct_draft.model_dump_json()
    
    # Calculate raw distance
    raw_distance = edit_distance(draft_json, correct_json)
    
    # Avoid division by zero if both are empty (unlikely but safe)
    max_len = max(len(draft_json), len(correct_json))
    if max_len == 0:
        return 1.0
        
    normalized_distance = raw_distance / max_len
    return 1.0 - normalized_distance

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
