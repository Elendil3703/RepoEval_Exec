import unittest
import json
import os
from typing import Sequence

TEST_RESULT_JSONL = "test_result.jsonl"

class TestSetGradsFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[267]  # Get the 268th JSON element (index 267)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON for set_grads functionality."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # To collect test results to be written as JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Ensure "set_grads" function exists in the code
                if "def set_grads" not in code:
                    print(f"Code snippet {i}: FAILED, function 'set_grads' not found.\n")
                    failed_count += 1
                    # Record failure
                    results.append({
                        "function_name": "set_grads",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'Sequence': Sequence,
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if 'set_grads' is in exec_locals
                    if 'set_grads' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'set_grads' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "set_grads",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Functionality test for 'set_grads'
                    class MockTensor:
                        def __init__(self):
                            self.grad = None

                    # Creating dummy params and grads
                    params = [MockTensor(), MockTensor()]
                    grads = [1, 2]

                    # Run the function
                    exec_locals['set_grads'](None, params, grads)

                    # Test if grads were set correctly
                    self.assertEqual(params[0].grad, 1, f"Code snippet {i} failed: first grad not set correctly.")
                    self.assertEqual(params[1].grad, 2, f"Code snippet {i} failed: second grad not set correctly.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "set_grads",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "set_grads",
                        "code": code,
                        "result": "failed"
                    })

        # Final test summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for set_grads
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "set_grads"
        ]

        # Append new results
        existing_records.extend(results)

        # Re-write the JSONL file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()