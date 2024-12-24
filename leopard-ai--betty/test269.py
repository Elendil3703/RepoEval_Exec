import unittest
import json
import os
import sys
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestZeroGradFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[268]  # Get the 269th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON for zero_grad function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                if "def zero_grad" not in code:
                    print(f"Code snippet {i}: FAILED, function 'zero_grad' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "zero_grad",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'sys': sys,
                    'Any': Any,  # Inject Any
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if zero_grad exists
                    if 'zero_grad' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'zero_grad' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "zero_grad",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Create a mock object to test zero_grad
                    class MockParameter:
                        def __init__(self):
                            self.grad = 1.0

                    class MockObject:
                        def trainable_parameters(self):
                            return [MockParameter() for _ in range(5)]

                    # Instantiate mock object and test zero_grad on it
                    mock_obj = MockObject()
                    exec_locals['zero_grad'](mock_obj)

                    # Verify all trainable parameters have their grad set to None
                    for param in mock_obj.trainable_parameters():
                        self.assertFalse(
                            hasattr(param, 'grad'),
                            f"Code snippet {i}: Parameter should not have 'grad' attribute."
                        )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "zero_grad",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "zero_grad",
                        "code": code,
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Append results to the existing test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Delete old records for function_name == "zero_grad"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "zero_grad"
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