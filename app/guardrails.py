from .models import DischargeSummary, CitedFact, MedicationChange
from typing import Dict, Any, List

def validate_citations(draft: DischargeSummary) -> List[str]:
    """
    Layer 2 Code-Level Guardrail.
    Iterates through the generated Pydantic model and ensures that any fact 
    not marked as 'MISSING', 'NOT_FOUND', or 'PENDING' has a valid source_citation.
    Returns a list of validation errors.
    """
    errors = []
    
    def check_fact(fact: CitedFact, field_name: str):
        val = fact.value.upper()
        if val not in ["MISSING", "NOT_FOUND", "PENDING", "UNDOCUMENTED"]:
            if not fact.source_citation or fact.source_citation.strip() == "":
                errors.append(f"Validation Error in '{field_name}': Fact '{fact.value}' has no source citation.")
                
    # Check top level fields
    check_fact(draft.patient_demographics, "patient_demographics")
    check_fact(draft.admission_date, "admission_date")
    check_fact(draft.discharge_date, "discharge_date")
    check_fact(draft.principal_diagnosis, "principal_diagnosis")
    check_fact(draft.hospital_course, "hospital_course")
    check_fact(draft.allergies, "allergies")
    check_fact(draft.follow_up_instructions, "follow_up_instructions")
    check_fact(draft.discharge_condition, "discharge_condition")
    
    # Check lists
    for i, diag in enumerate(draft.secondary_diagnoses):
        check_fact(diag, f"secondary_diagnoses[{i}]")
        
    for i, proc in enumerate(draft.procedures):
        check_fact(proc, f"procedures[{i}]")
        
    for i, pending in enumerate(draft.pending_results):
        check_fact(pending, f"pending_results[{i}]")
        
    # Check medications
    for i, med in enumerate(draft.discharge_medications):
        check_fact(med.reason_for_change, f"discharge_medications[{i}].reason_for_change")
        
    return errors
