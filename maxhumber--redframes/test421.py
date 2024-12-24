import unittest
import json
import os
import sys
import pandas as pd
from pandas import DataFrame as PandasDataFrame

TEST_RESULT_JSONL = "test_result.jsonl"

class TestAppendFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[420]  # Get the 421st JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 421st JSON array")

    def test_append_function(self):
        """Dynamically test all code snippets for 'append' function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to be written to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                if "def append" not in code:
                    print(f"Code snippet {i}: FAILED, function 'append' not found.\n")
                    failed_count += 1
                    # Write failure record
                    results.append({
                        "function_name": "append",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'pd': pd,
                    'PandasDataFrame': PandasDataFrame
                }
                exec_locals = {}

                try:
                    # Dynamically execute code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if append function exists
                    if 'append' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'append' function not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "append",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test append function with sample data
                    top_df = PandasDataFrame({'A': [1, 2], 'B': [3, 4]})
                    bottom_df = PandasDataFrame({'A': [5, 6], 'B': [7, 8]})
                    expected_df = pd.concat([top_df, bottom_df]).reset_index(drop=True)

                    result_df = exec_locals['append'](top_df, bottom_df)

                    # Perform assertions
                    pd.testing.assert_frame_equal(expected_df, result_df)

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "append",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "append",
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

        # Remove old records with function_name == "append"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "append"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()