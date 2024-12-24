import unittest
import json
import os
import torch
import torch.nn.functional as F
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestTrainingStepFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[293]  # Get the 294th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_training_step(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect the results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Check if the function training_step is defined
                if "def training_step" not in code:
                    print(f"Code snippet {i}: FAILED, function 'training_step' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "training_step",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'torch': torch,
                    'F': F,
                    'Any': Any
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if 'training_step' really exists in the locals
                    self.assertIn('training_step', exec_locals)

                    class DummyModule:
                        def __call__(self, inputs):
                            return inputs * 2, torch.ones((inputs.size(0), 1))

                    class TestClass:
                        def __init__(self):
                            self.module = DummyModule()
                        
                        def outer(self):
                            return torch.tensor([1.0])

                    # Test the training_step function
                    instance = TestClass()
                    batch = (torch.tensor([0.5]), torch.tensor([1.0]))

                    # Capture the loss
                    loss = exec_locals['training_step'](instance, batch)
                    
                    # Assert loss is a tensor
                    self.assertIsInstance(loss, torch.Tensor, "Loss is not a tensor.")
                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "training_step",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "training_step",
                        "code": code,
                        "result": "failed"
                    })

        # Test summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for "training_step"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "training_step"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite the JSONL file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()