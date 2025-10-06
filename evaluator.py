import subprocess
import sys
import os
from datetime import datetime

class Evaluator:
    def __init__(self, methods, dataset_path="data/ground-truth/QA-Dataset.csv", output_dir="./outputs"):
        self.methods = methods
        self.dataset_path = dataset_path
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = os.path.join(output_dir, self.timestamp)
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate(self, onlyhead:bool = False):
        for method in self.methods:
            print(f"\n🚀 Running evaluation for: {method}\n")

            process = subprocess.Popen(
                [sys.executable, "evaluate_method.py", method, self.dataset_path, self.output_dir, str(onlyhead)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            # Stream output live
            for line in process.stdout:
                print(line, end="")  # print each line as it comes

            process.wait()  # wait until subprocess finishes

            if process.returncode != 0:
                print(f"⚠ Method '{method}' failed with return code {process.returncode}.\n")

        print(f"\n✅ Evaluation finished.\n")

if __name__ == "__main__":
    methods = ["semantic", "recursive", "doublepass"]
    evaluator = Evaluator(methods)
    summary = evaluator.evaluate()
