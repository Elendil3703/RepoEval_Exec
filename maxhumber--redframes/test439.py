import unittest
import json
import os
import sys
from typing import List, Union
import pandas as pd

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGroupFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[438]  # Get the 439th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in code_snippets")

    def test_group_function(self):
        """Dynamically test the group function in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # To collect test results for JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Static check: Ensure the function 'group' is defined
                if "def group" not in code:
                    print(f"Code snippet {i}: FAILED, function 'group' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "group",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Execute the code snippet
                exec_globals = {
                    'pd': pd,
                    'sys': sys
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Ensure 'group' function is available
                    if 'group' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'group' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "group",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    group = exec_locals['group']

                    # Test data
                    df = pd.DataFrame({'a': [1, 2, 1, 2], 'b': ['x', 'y', 'x', 'y'], 'c': [10, 20, 30, 40]})

                    # Test the group function
                    grouped = group(df, by='a')
                    self.assertIsInstance(grouped, pd.core.groupby.DataFrameGroupBy, f"Code snippet {i} did not return a DataFrameGroupBy object.")
                    self.assertEqual(len(list(grouped)), 2, f"Code snippet {i} returned incorrect number of groups.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "group",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "group",
                        "code": code,
                        "result": "failed"
                    })

        # Final statistics
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

        # Remove old records with function_name == "group"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "group"
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