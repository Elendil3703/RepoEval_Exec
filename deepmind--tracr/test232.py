import unittest
import json
import sys
import os
from typing import Any, Callable
import functools
import logging

TEST_RESULT_JSONL = "test_result.jsonl"

logging.basicConfig(level=logging.WARNING)

class TestIgnoringArithmeticErrors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[231]  # Get the 232nd JSON element (index 231)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 232nd JSON array")

    def test_ignoring_arithmetic_errors(self):
        """Dynamically test all code snippets for ignoring_arithmetic_errors."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to the JSONL file

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Check if the necessary function exists in the code snippet
                if "def ignoring_arithmetic_errors" not in code:
                    print(f"Code snippet {i}: FAILED, 'ignoring_arithmetic_errors' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "ignoring_arithmetic_errors",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'functools': functools,
                    'logging': logging,
                    'Callable': Callable,
                    'Any': Any
                }
                exec_locals = {}

                try:
                    # Execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if ignoring_arithmetic_errors is available
                    if 'ignoring_arithmetic_errors' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'ignoring_arithmetic_errors' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "ignoring_arithmetic_errors",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test the ignoring_arithmetic_errors decorator
                    def division(a, b):
                        return a / b

                    safe_division = exec_locals['ignoring_arithmetic_errors'](division)

                    # Check normal operation
                    self.assertEqual(safe_division(4, 2), 2, f"Code snippet {i} failed on normal division.")

                    # Check error handling
                    self.assertIsNone(safe_division(4, 0), f"Code snippet {i} failed on division by zero.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "ignoring_arithmetic_errors",
                        "code": code,
                        "result": "passed"
                    })

                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "ignoring_arithmetic_errors",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
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

        # Remove old records for this function
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "ignoring_arithmetic_errors"
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