import json
import os
from collections import Counter
from typing import List
from app.config import MEMORY_FILE_PATH, TRAIN_TOP_K_MISTAKES

class CorrectionMemory:
    def __init__(self, memory_file: str = MEMORY_FILE_PATH):
        self.memory_file = memory_file
        self.mistake_frequencies = Counter()
        self._load()
        
    def _load(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                    self.mistake_frequencies = Counter(data)
            except Exception:
                pass
                
    def _save(self):
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        with open(self.memory_file, 'w') as f:
            json.dump(dict(self.mistake_frequencies), f, indent=2)
            
    def add_corrections(self, reasons: List[str]):
        """Adds new correction reasons to the memory."""
        # For simplicity, we just count the exact string frequencies. 
        # In a real system, you might embed these and cluster them to count semantic frequencies.
        for reason in reasons:
            self.mistake_frequencies[reason] += 1
        self._save()
        
    def get_context(self, top_k: int = 5) -> str:
        """Returns a string formatted with the top K mistakes to inject into the agent's prompt."""
        print(f"Top k to inject: {top_k}")
        if not self.mistake_frequencies:
            return ""
            
        most_common = self.mistake_frequencies.most_common(top_k)
        context = []
        for i, (mistake, count) in enumerate(most_common, 1):
            context.append(f"{i}. {mistake} (seen {count} times)")
            
        return "\n".join(context)
