import os
import json
import matplotlib.pyplot as plt
from app.agent import DischargeSummaryAgent
from correction_agent.reviewer import SimulatedReviewer
from correction_agent.memory import CorrectionMemory
from correction_agent.metrics import calculate_schema_edit_score, calculate_schema_edit_distance, calculate_flag_recall
from correction_agent.synthetic_data import generate_synthetic_patients
from app.tools import read_pdf_pages
from app.config import SYNTHETIC_DATA_PATHS, TEST_PDF_PATH, MEMORY_FILE_PATH, TRAIN_TOP_K_MISTAKES, TEST_TOP_K_MISTAKES
from dotenv import load_dotenv
from app.models import DischargeSummary

load_dotenv()

def read_json_source(json_path: str) -> str:
    """Reads a synthetic JSON patient file and returns the clinical text string."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data.get("text", str(data))

def main():
    agent = DischargeSummaryAgent()
    reviewer = SimulatedReviewer()
    memory = CorrectionMemory(MEMORY_FILE_PATH)
    
    # 1. Ensure synthetic data exists
    synthetic_files = SYNTHETIC_DATA_PATHS
    
    missing_files = [p for p in synthetic_files if not os.path.exists(p)]
    if missing_files:
        print("Synthetic JSON files not found. Generating them now...")
        generate_synthetic_patients()
    
    test_pdf = TEST_PDF_PATH
    
    edit_scores = []
    edit_distances = []
    x_labels = []
    
    # 2. Training Loop
    print("=== STARTING LEARNING LOOP ===")
    for iteration, json_path in enumerate(synthetic_files, 1):
        print(f"\n--- Iteration {iteration} on {json_path} ---")
        
        # Get memory context
        context = memory.get_context(top_k=TRAIN_TOP_K_MISTAKES)
        if context:
            print(f"Injecting Correction Context:\n{context}")
        
        # Load JSON source text directly (no OCR needed)
        source_text = read_json_source(json_path)
            
        # Agent generates draft, with source text pre-injected
        #draft, trace = agent.run(pdf_path=json_path, corrections_context=context, source_text=source_text)
        with open("final_discharge_summary.json", "r") as f:
            draft = DischargeSummary.model_validate_json(f.read())
        if not draft:
            print("Agent failed to generate draft.")
            continue

        with open("agent_trace.json", "r") as f:
            trace = json.load(f)
            
        # Reviewer checks draft using the same pre-loaded source text
        print("\nSimulated Reviewer evaluating draft...")
        corrected_draft, reasons = reviewer.review_draft(draft, source_text)
        
        # Metrics
        score = calculate_schema_edit_score(draft, corrected_draft)
        distance = calculate_schema_edit_distance(draft, corrected_draft)
        recall = calculate_flag_recall(draft, corrected_draft)
        
        edit_scores.append(score)
        edit_distances.append(distance)
        x_labels.append(f"Train {iteration}")
        
        print(f"Metrics -> Edit Distance: {distance}, Accuracy Score: {score:.3f}, Safety Recall: {recall:.2f}")
        print(f"Correction Reasons Identified: {reasons}")
        
        # Update Memory
        if reasons:
            memory.add_corrections(reasons)
            
    # 3. Validation on Held-Out Test Set (real PDF via OCR)
    print(f"\n=== VALIDATING ON HELD-OUT TEST SET ({test_pdf}) ===")
    context = memory.get_context(top_k=TEST_TOP_K_MISTAKES)
    print(f"Final Trained Context:\n{context}")
    
    final_draft, trace = agent.run(pdf_path=test_pdf, corrections_context=context)
    # with open("final_discharge_summary.json", "r") as f:
    #     final_draft = DischargeSummary.model_validate_json(f.read())
        
    if final_draft:
        source_text = read_pdf_pages(test_pdf)  # OCR for the real PDF
        corrected_draft, reasons = reviewer.review_draft(final_draft, source_text)
        score = calculate_schema_edit_score(final_draft, corrected_draft)
        distance = calculate_schema_edit_distance(final_draft, corrected_draft)
        recall = calculate_flag_recall(final_draft, corrected_draft)
        
        edit_scores.append(score)
        edit_distances.append(distance)
        x_labels.append("Test (Real Data)")
        print(f"Final Test Metrics -> Edit Distance: {distance}, Accuracy Score: {score:.3f}, Safety Recall: {recall:.2f}")
    
    # 4. Plot Learning Curves
    if not edit_scores:
        print("No valid metrics gathered to plot (likely due to API limits).")
        return
    
    # Plot 1: Accuracy Score
    plt.figure(figsize=(8, 5))
    plt.plot(x_labels, edit_scores, marker='o', linestyle='-', color='g')
    plt.title('Agent Improvement: Accuracy Score over Iterations')
    plt.ylabel('Accuracy Score (Higher is Better, Max 1.0)')
    plt.xlabel('Evaluation Phase')
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.savefig('reward_learning_curve.png')
    
    # Plot 2: Raw Edit Distance
    plt.figure(figsize=(8, 5))
    plt.plot(x_labels, edit_distances, marker='o', linestyle='-', color='b')
    plt.title('Agent Improvement: Raw Edit Distance over Iterations')
    plt.ylabel('Edit Distance (Lower is Better)')
    plt.xlabel('Evaluation Phase')
    plt.grid(True)
    plt.savefig('learning_curve.png')
    
    print("\nSaved reward_learning_curve.png and learning_curve.png")

if __name__ == "__main__":
    main()
