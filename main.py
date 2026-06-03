import os
import json
from app.agent import DischargeSummaryAgent
from app.config import GEMINI_API_KEY

def main():
    print("🚀 Initializing Discharge Summary Agent...")
    
    try:
        agent = DischargeSummaryAgent()
        
        print("\n⏳ Agent is now running the ReAct loop on the patient data...")
        draft, trace = agent.run()
        
        if draft:
            output_file = "final_discharge_summary.json"
            with open(output_file, "w") as f:
                f.write(draft.model_dump_json(indent=4))
            print(f"\n✅ Success! Final draft saved to {output_file}")
        else:
            print("\n❌ Agent failed to generate a complete draft.")
            
        trace_file = "agent_trace.json"
        with open(trace_file, "w") as f:
            json.dump(trace, f, indent=4)
        print(f"✅ Full execution trace saved to {trace_file}")
        
    except Exception as e:
        print(f"\n❌ Error starting agent: {e}")

if __name__ == "__main__":
    main()
