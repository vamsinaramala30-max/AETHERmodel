"""
AETHER MODEL — Model Evaluation & Benchmark Execution
Evaluates trained checkpoint across benchmark evaluation dataset and assigns quality classification.
"""

import os
import sys
import math
from typing import Dict, Any, Optional

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(base_dir, "src")
for p in [base_dir, src_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model.model import AetherModel
from tokenizer.tokenizer import AetherTokenizer
from training.quality_gate import QualityGate

class ModelEvaluator:
    def __init__(self, model_checkpoint: Optional[str] = None):
        self.model = AetherModel()
        if model_checkpoint and os.path.exists(model_checkpoint):
            self.model.load_checkpoint(model_checkpoint)
        self.tokenizer = AetherTokenizer()
        self.quality_gate = QualityGate(model=self.model, tokenizer=self.tokenizer)

    def evaluate_suite(self, eval_suite_path: Optional[str] = None) -> Dict[str, Any]:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        eval_path = eval_suite_path or os.path.join(base_dir, "data", "evaluation", "aether_eval_suite.jsonl")
        return self.quality_gate.evaluate_suite(eval_path)

    def evaluate_sequence(self, text: str) -> Dict[str, float]:
        ids = self.tokenizer.encode(text)
        if len(ids) < 2:
            return {"loss": 0.0, "perplexity": 1.0}

        total_loss = 0.0
        for i in range(len(ids) - 1):
            input_seq = ids[:i + 1]
            target_id = ids[i + 1]
            logits = self.model.forward(input_seq)

            max_l = max(logits)
            exps = [math.exp(l - max_l) for l in logits]
            sum_e = sum(exps) or 1e-9
            prob = max(exps[target_id % len(exps)] / sum_e, 1e-9)
            total_loss += -math.log(prob)

        avg_loss = total_loss / (len(ids) - 1)
        perplexity = math.exp(min(avg_loss, 20.0))
        return {"loss": round(avg_loss, 4), "perplexity": round(perplexity, 4)}

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    report = evaluator.evaluate_suite()
    print(f"[AETHER EVALUATION] Quality Gate: {report['quality_classification']} | Accuracy: {report['accuracy'] * 100}% ({report['passed_tests']}/{report['total_tests']})")
