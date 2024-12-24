import unittest
import json
import sys
import os
import math
import torch
from einops import rearrange

TEST_RESULT_JSONL = "test_result.jsonl"

class TestForwardFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file and extract required code
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[72]  # Get the 73rd JSON element (0-indexed)

    def test_forward(self):
        """Test the forward function defined in the code snippet."""
        passed_count = 0
        failed_count = 0
        results = []

        code = self.code_snippet
        with self.subTest(code=code):
            print("Testing forward function...")

            # Prepare environment for execution
            exec_globals = {
                'math': math,
                'torch': torch,
                'rearrange': rearrange,
            }
            exec_locals = {}

            try:
                # Dynamically execute the code snippet
                exec(code, exec_globals, exec_locals)

                # Check if 'forward' function is defined
                if 'forward' not in exec_locals:
                    print("Code snippet: FAILED, 'forward' function not found.")
                    failed_count += 1
                    results.append({
                        "function_name": "forward",
                        "code": code,
                        "result": "failed"
                    })
                    return

                forward_func = exec_locals['forward']
                
                # Define some test cases
                class Dummy:
                    def __init__(self, weights):
                        self.weights = weights

                weights = torch.tensor([1.0, 2.0])
                dummy_instance = Dummy(weights=weights)

                test_input = torch.tensor([3.0])
                expected_output_shape = (1, 5)

                # Call the forward function
                try:
                    output = forward_func(dummy_instance, test_input)
                    self.assertEqual(
                        output.shape,
                        expected_output_shape,
                        "Output shape mismatch."
                    )
                    print("Code snippet: PASSED all assertions.")
                    passed_count += 1
                    results.append({
                        "function_name": "forward",
                        "code": code,
                        "result": "passed"
                    })

                except Exception as e:
                    print(f"Code snippet: FAILED with error during forward call: {e}.")
                    failed_count += 1
                    results.append({
                        "function_name": "forward",
                        "code": code,
                        "result": "failed"
                    })

            except Exception as e:
                print(f"Code snippet: FAILED with error: {e}")
                failed_count += 1
                results.append({
                    "function_name": "forward",
                    "code": code,
                    "result": "failed"
                })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed\n")
        
        # ============= Write results to test_result.jsonl =============
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