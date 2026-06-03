from pydantic import BaseModel, Field
from typing import List, Optional, Any

class CitedFact(BaseModel):
    value: str = Field(description="The extracted fact. Use 'MISSING' or 'NOT_FOUND' if not present in the document.")
    source_citation: str = Field(description="The document name and page number where this fact was found. e.g., 'patient_report.pdf, Page 1'. Leave empty ONLY if value is 'MISSING' or 'NOT_FOUND'.")

class MedicationChange(BaseModel):
    medication_name: str
    change_type: str = Field(description="One of: 'ADDED', 'STOPPED', 'CHANGED', 'UNCHANGED'")
    reason_for_change: CitedFact = Field(description="Why the medication was changed. Must have a citation. If no reason is documented, flag it as 'UNDOCUMENTED'.")

class DischargeSummary(BaseModel):
    patient_demographics: CitedFact
    admission_date: CitedFact
    discharge_date: CitedFact
    principal_diagnosis: CitedFact
    secondary_diagnoses: List[CitedFact] = Field(default_factory=list)
    hospital_course: CitedFact
    procedures: List[CitedFact] = Field(default_factory=list)
    discharge_medications: List[MedicationChange] = Field(default_factory=list)
    allergies: CitedFact
    follow_up_instructions: CitedFact
    pending_results: List[CitedFact] = Field(default_factory=list, description="Any labs or tests pending at discharge.")
    discharge_condition: CitedFact
    
    conflicts_flagged: List[str] = Field(default_factory=list, description="List of conflicting information found across notes.")
    escalations: List[str] = Field(default_factory=list, description="List of issues escalated for clinician review (e.g., undocumented medication changes, critical drug interactions).")
