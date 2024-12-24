import unittest
import json
import os
import torch.nn as nn  # Import PyTorch's nn module
from torch.nn import Module
from torch.nn.modules import Linear
from torch import tensor
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestZeroInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file and extract the specified code snippet
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[54]  # Get the 55th JSON element (index 54)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 55th JSON array")

    def test_zero_init(self):
        """Dynamically test all code snippets to ensure zero_init_ initializes weights and biases to zeros."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to be written to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Define exec environment
                exec_globals = {
                    'nn': nn,  # Inject nn module for init functions
                    'Module': Module,  # Base class for PyTorch modules
                    'Linear': Linear,  # Example layer to test
                    'tensor': tensor,  # For creating tensors
                    'Any': Any,
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Ensure zero_init_ function is defined
                    if 'zero_init_' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'zero_init_' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "zero_init_",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    zero_init_ = exec_locals['zero_init_']

                    # Test the function using a Linear layer, which has both weights and biases
                    layer = Linear(10, 5)
                    zero_init_(layer)  # Apply zero initialization

                    # Assertions
                    self.assertTrue(
                        torch.equal(layer.weight, torch.zeros_like(layer.weight)),
                        f"Code snippet {i}: Weights were not initialized to zero."
                    )

                    if layer.bias is not None:
                        self.assertTrue(
                            torch.equal(layer.bias, torch.zeros_like(layer.bias)),
                            f"Code snippet {i}: Biases were not initialized to zero."
                        )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "zero_init_",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "zero_init_",
                        "code": code,
                        "result": "failed"
                    })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write test results to test_result.jsonl =============
        # Read existing test_result.jsonl (if it exists)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for the zero_init_ function
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "zero_init_"
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