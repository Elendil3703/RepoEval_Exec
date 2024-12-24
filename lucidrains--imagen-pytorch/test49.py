import unittest
import json
import sys
import re
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestInnerFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[48]  # Retrieve the 49th JSON element (index 48)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 49th JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets for the 'inner' function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Check for the definition of 'inner'
                if "def inner" not in code:
                    print(f"Code snippet {i}: FAILED, 'inner' function not defined in code.\n")
                    failed_count += 1
                    # Write the failure record
                    results.append({
                        "function_name": "inner",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Pattern for matching the 'inner' function signature
                func_pattern = r"def\s+inner\s*\("
                if not re.search(func_pattern, code):
                    print(f"Code snippet {i}: FAILED, incorrect signature for 'inner'.\n")
                    failed_count += 1
                    # Write the failure record
                    results.append({
                        "function_name": "inner",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Dynamic execution and testing logic
                exec_globals = {
                    'sys': sys,
                    'Any': Any,
                    'exists': lambda x: x is not None,
                    'fn': lambda x: x * 2
                }
                exec_locals = {}

                try:
                    # Execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if 'inner' was added to exec_locals
                    if 'inner' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'inner' function not found after execution.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "inner",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Get the 'inner' function
                    inner_func = exec_locals['inner']

                    # Perform tests on the 'inner' function
                    self.assertEqual(inner_func(None), None,
                                     f"Code snippet {i} failed, expected inner(None) to return None.")
                    self.assertEqual(inner_func(2), 4,
                                     f"Code snippet {i} failed, expected inner(2) to return 4.")
                    self.assertEqual(inner_func(0), 0,
                                     f"Code snippet {i} failed, expected inner(0) to return 0.")
                    
                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "inner",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "inner",
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

        # Remove existing records for the 'inner' function
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "inner"
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