import json
import os
from google import genai
from .prompts import get_system_prompt
from .config import GEMINI_API_KEY, AGENT_MODEL, MAX_STEPS, USE_CRITIC_AGENT
from .tools import execute_tool
from .models import DischargeSummary
from .guardrails import validate_citations
from .critic import CriticAgent

class DischargeSummaryAgent:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("GEMINI_KEY is missing.")
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = AGENT_MODEL
        self.trace_log = []
        self.max_steps = MAX_STEPS
        
    def log_trace(self, step: int, thought: str, action: str, action_input: str, result: str):
        self.trace_log.append({
            "step": step,
            "thought": thought,
            "action": action,
            "action_input": action_input,
            #"result": result[:500] + "..." if len(result) > 500 else result # Truncate long results for trace readability
            "result": result
        })
        print(f"\n[STEP {step}]")
        print(f"THOUGHT: {thought}")
        print(f"ACTION: {action} ({action_input})")
        print(f"RESULT: {result[:200]}...\n")

    def run(self, pdf_path: str = "docs/patient_report.pdf", corrections_context: str = "", source_text: str = "") -> tuple[DischargeSummary | None, list]:
        """Runs the ReAct loop until completion or MAX_STEPS.
        
        Args:
            pdf_path: Path to the source document (PDF for real patient, JSON for synthetic).
            corrections_context: Top-K mistakes from CorrectionMemory to inject into the prompt.
            source_text: If provided (e.g. pre-loaded JSON text), this is injected directly into
                         the conversation so the agent doesn't need to call the read_pdf_pages tool.
        """
        step_count = 0
        print(f"Max steps: {self.max_steps}")
        
        # History tracks the conversation context
        history = [{"role": "user", "parts": [{"text": get_system_prompt(pdf_path, corrections_context)}]}]
        
        # If source text is pre-loaded (e.g. from a JSON file), inject it directly.
        # This bypasses the read_pdf_pages OCR tool call.
        if source_text:
            history.append({"role": "user", "parts": [{"text": f"Source Document Content:\n{source_text}\n\nNow proceed with your analysis and use the remaining tools (drug_interaction_check, escalate) as needed, then COMPOSE the final summary."}]})
        
        while step_count < self.max_steps:
            print(f"Current step: {step_count}")
            step_count += 1
            
            # 1. Ask the model for the next action
            try:
                # We ask the model to reply in strict JSON
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=history,
                    config=genai.types.GenerateContentConfig(
                        response_mime_type="application/json",
                    )
                )
                
                response_text = response.text
                
                try:
                    action_data = json.loads(response_text)
                    thought = action_data.get("thought", "")
                    action = action_data.get("action", "")
                    action_input = action_data.get("action_input", "")
                except json.JSONDecodeError:
                    error_msg = "Error: Failed to parse your response as JSON. Please ensure you output strictly valid JSON matching the requested structure."
                    self.log_trace(step_count, "JSON parsing failed", "ERROR", "", error_msg)
                    history.append({"role": "model", "parts": [{"text": response_text}]})
                    history.append({"role": "user", "parts": [{"text": error_msg}]})
                    continue
                
                # 2. Execute Action
                if action == "COMPOSE":
                    self.log_trace(step_count, thought, action, action_input, "Proceeding to generation phase.")
                    break # Break out of ReAct loop, ready to generate final schema
                
                tool_result = execute_tool(action, action_input, pdf_path=pdf_path)
                self.log_trace(step_count, thought, action, action_input, tool_result)
                
                # 3. Append to history for next iteration
                history.append({"role": "model", "parts": [{"text": response_text}]})
                history.append({"role": "user", "parts": [{"text": f"Tool Output: {tool_result}"}]})
                
            except Exception as e:
                self.log_trace(step_count, "Exception during generation", "ERROR", "", str(e))
                history.append({"role": "user", "parts": [{"text": f"Error occurred: {str(e)}. Please retry or formulate a new plan."}]})

        if step_count >= self.max_steps:
            print("\n⚠️ WARNING: MAX_STEPS reached without COMPOSE.")
            
        # 4. Generate the final structured Draft
        print("\n📝 Composing Final Draft...")
        compose_prompt = "You have finished your investigation. Generate the final DischargeSummary based on your findings."
        history.append({"role": "user", "parts": [{"text": compose_prompt}]})
        
        try:
            final_response = self.client.models.generate_content(
                model=self.model_name,
                contents=history,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=DischargeSummary, # Let the Gemini SDK handle Pydantic schema enforcing
                )
            )
            
            draft_data = json.loads(final_response.text)
            draft = DischargeSummary(**draft_data)
            
            # 5. Run Layer 2 Guardrails
            validation_errors = validate_citations(draft)
            if validation_errors:
                print("\n🚨 GUARDRAIL VALIDATION ERRORS FOUND:")
                for err in validation_errors:
                    print(f" - {err}")
                print("The agent attempted to fabricate data without a citation!")
            
            
            # 5. Run Critic Agent Self-Reflection
            if USE_CRITIC_AGENT:
                print("\n🧐 Critic Agent reflecting on draft...")
                critic = CriticAgent(api_key=self.api_key)
                # If source_text was pre-loaded, pass it directly; else critic reads the PDF itself
                draft = critic.reflect(draft, pdf_path=pdf_path, source_text=source_text)
            else:
                print("\n⏩ Critic Agent disabled (USE_CRITIC_AGENT = False). Skipping reflection...")
            
            return draft, self.trace_log
            
        except Exception as e:
            print(f"Failed to generate final compose: {e}")
            return None, self.trace_log
