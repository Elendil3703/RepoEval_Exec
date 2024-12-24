import unittest
import json
import os
import torch
import torch.nn.functional as F
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestForwardFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[380]  # Get the 381st JSON element (index 380)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 381st JSON array")

    def test_forward_function(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Static check: Verify the presence of 'def forward' in the code snippet
                if "def forward" not in code:
                    print(f"Code snippet {i}: FAILED, 'def forward' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "forward",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'torch': torch,
                    'F': F,
                    'Any': Any,
                }
                exec_locals = {}

                try:
                    exec(code, exec_globals, exec_locals)

                    # Ensure 'forward' function exists
                    if 'forward' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'forward' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "forward",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Mock class and invoke the forward function
                    class MockModel:
                        def __init__(self, target_dim):
                            self.target_dim = target_dim
                            self.forward = exec_locals['forward'].__get__(self)

                    model = MockModel(target_dim=10)
                    
                    # Generate sample input
                    input_features = torch.rand(3, 30) # batchsize=3, num_features=30

                    # Call the forward function
                    output = model.forward(input_features)

                    # Assertions: The shape of output should be (batchsize, target_dim)
                    self.assertEqual(
                        output.shape,
                        (3, 10),
                        f"Code snippet {i} output has incorrect shape: {output.shape}. Expected: (3, 10).",
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "forward",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "forward",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Writing results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for function_name == "forward"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "forward"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()