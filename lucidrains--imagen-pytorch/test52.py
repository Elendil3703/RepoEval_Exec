import unittest
import json
import sys
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

def exists(val):
    return val is not None

class TestDefaultFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[51]  # Get the 52nd JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 52nd JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets for the 'default' function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results for JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                # Check if 'default' function is defined in the snippet
                if "default" not in code:
                    print(f"Code snippet {i}: FAILED, 'default' function not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "default",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'exists': exists,
                    'Any': Any,
                }
                exec_locals = {}

                try:
                    # Execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Verify 'default' function is available
                    if 'default' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'default' function not in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "default",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    default_fn = exec_locals['default']

                    # Test cases for 'default' function
                    test_cases = [
                        (None, lambda: 5, 5),
                        ('value', lambda: 10, 'value'),
                        (None, 3, 3),
                        ('value', 3, 'value'),
                        (None, 'fallback', 'fallback'),
                    ]

                    for val, d, expected in test_cases:
                        with self.subTest(val=val, d=d):
                            result = default_fn(val, d)
                            self.assertEqual(result, expected, f"Failed for val={val}, d={d}")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "default",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "default",
                        "code": code,
                        "result": "failed"
                    })

        # Summary report
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

        # Remove old records for 'default' function
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "default"
        ]

        # Append new results
        existing_records.extend(results)

        # Write back to the JSONL file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()