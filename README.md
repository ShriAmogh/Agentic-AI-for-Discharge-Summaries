# Agentic AI for Discharge Summaries (DScribe Assignment)

This repository contains an AI-powered pipeline designed to automate the extraction and synthesis of highly accurate, clinically safe Discharge Summaries from unstructured hospital records. The system is split into two phases: a robust **ReAct Agent** (Part 1) and an **Agentic Learning Loop** (Part 2) that allows the agent to learn from its past mistakes over time.

## Architecture & Agents

### Part 1: The Core Pipeline & Safety Guardrails
The foundation of the system is a medical AI assistant operating in a Reason-Act (ReAct) loop.

- **DischargeSummaryAgent (`app/agent.py`)**: The primary agent. It is given a set of tools (PDF reader, drug interaction checker, clinician escalation) and iterates through the documents to gather facts. It strictly enforces citations to prevent hallucination.
- **CriticAgent (`app/critic.py`)**: A critical safety module. Before the main agent finalizes its draft, the CriticAgent runs a self-reflection pass over the draft and the source text. It aggressively checks for unescalated medication changes or conflicting diagnoses and intervenes if the draft is unsafe.

### Part 2: The Agentic Learning Loop (Continuous Improvement)
To improve performance and safety over time, we wrap the base agent in a training loop that simulates senior clinician feedback.

- **SimulatedReviewer (`correction_agent/reviewer.py`)**: Acts as a strict attending physician. It compares the agent's draft against the raw source text, identifies structural errors (missing labs, fabricated facts), outputs a corrected "Perfect Draft," and generates specific "Correction Reasons."
- **CorrectionMemory (`correction_agent/memory.py`)**: A persistent memory module that tracks the frequency of the agent's mistakes. It aggregates the most common errors and injects them back into the `DischargeSummaryAgent`'s system prompt in subsequent iterations, forcing the agent to avoid those specific pitfalls.
- **Synthetic Data Generator (`correction_agent/synthetic_data.py`)**: An LLM-powered script that generates highly realistic, complex, and messy clinical charts (JSON format) used to train the agent in the learning loop.

## How to Run

### Prerequisites
1. Ensure you have Python 3.11+ and a virtual environment (`.venv`) set up.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and add your Gemini API key:
   ```env
   GEMINI_KEY="your_api_key_here"
   ```

### Running Part 1 (Single Patient Generation)
To run the standard agent on the provided `docs/patient_report.pdf` without the learning loop:
```bash
python main.py
```
**Control Flow (`main.py`)**:
1. Initializes the `DischargeSummaryAgent`.
2. The agent enters a ReAct (Reason-Act) loop, reading the source PDF using tools and gathering clinical facts.
3. Once all facts are gathered, it generates a draft JSON summary.
4. The `CriticAgent` reviews the draft and intervenes if critical escalations are missing.
5. The final output is printed and saved to disk.

**Data Saved**:
- `final_discharge_summary.json` (Root directory): The final structured Pydantic schema output.
- `agent_trace.json` (Root directory): A step-by-step log of the agent's thought process, tool calls, and results.

### Running Part 2 (The Learning Loop)
To execute the self-improving training loop:
```bash
python train.py
```
**Control Flow (`train.py`)**:
1. **Data Generation**: Checks for existing synthetic data. If absent, it invokes `synthetic_data.py` to prompt Gemini to generate complex JSON clinical charts for 3 synthetic patients.
2. **Training Iterations**: For each synthetic patient:
   - Fetches the top mistakes from `CorrectionMemory` and injects them into the agent's prompt.
   - `DischargeSummaryAgent` generates a draft summary.
   - `CriticAgent` performs a safety reflection pass.
   - `SimulatedReviewer` compares the draft against the raw source text, outputting a corrected draft and specific mistake reasons.
   - Calculates Edit Distance and Safety Recall metrics.
   - Updates the `CorrectionMemory` with the new mistakes.
3. **Validation**: The fully-trained agent (loaded with max memory context) is evaluated on the highly complex, held-out test set (`docs/patient_report.pdf`).
4. **Visualization**: Plots the Schema Edit Distance across all iterations to visualize agent improvement.

**Data Saved**:
- `docs/synthetic_data/synthetic_patient_X.json`: The highly realistic, generated clinical charts used for training.
- `docs/memory/correction_memory.json`: The persistent frequency dictionary of all mistakes the agent has made.
- `learning_curve.png` (Root directory): A matplotlib line chart demonstrating the change in Edit Distance over the training and testing iterations.
4. **Metrics**: It plots the Schema Edit Distance across the iterations and saves it as `learning_curve.png`, visualizing the agent's improvement.

## Configuration
All model selections, iteration limits, file paths, and memory settings are centralized in `app/config.py`. You can adjust `MAX_STEPS`, `MAX_PAGES`, or swap out different Gemini models directly from this file.
