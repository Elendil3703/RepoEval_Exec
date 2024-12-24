import unittest
import json
import sys
import os
from functools import wraps  # 确保注入的环境中有 wraps
from typing import Any  # 注入必要的类型

TEST_RESULT_JSONL = "test_result.jsonl"

class TestOnceDecorator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[91]  # Get the 92nd JSON element
        if not cls.code_snippets:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_once_decorator(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results for writing to the JSONL file

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # ------------------- Perform Static Analysis -------------------
                if "def once" not in code:
                    print(f"Code snippet {i}: FAILED, function 'once' not defined in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "once",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Execute and Test Logic -------------------
                exec_globals = {
                    'wraps': wraps,
                    'Any': Any,  # Inject Any
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check whether 'once' really exists in the executed locals
                    if 'once' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'once' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "once",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test the once decorator
                    @exec_locals['once']
                    def sample_function(x):
                        return x * 2

                    # Call the function once and then again
                    first_call = sample_function(10)
                    second_call = sample_function(20)

                    # Validate the function was executed only once
                    self.assertEqual(first_call, 20, f"Code snippet {i} function executed with incorrect result.")
                    self.assertIsNone(second_call, f"Code snippet {i} function was not supposed to execute again.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "once",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "once",
                        "code": code,
                        "result": "failed"
                    })

        # Print final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write the test results to test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for the function 'once'
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "once"
        ]

        # Append new results
        existing_records.extend(results)

        # Overwrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()