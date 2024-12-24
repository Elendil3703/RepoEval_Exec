import unittest
import json
import pandas as pd
import os
from typing import Any, Dict, Tuple, Union

TEST_RESULT_JSONL = "test_result.jsonl"

class PandasDataFrame(pd.DataFrame):
    pass

class PandasGroupedFrame(pd.core.groupby.DataFrameGroupBy):
    pass

Column = str
Func = Any

def _check_type(obj, expected_type):
    if not isinstance(obj, expected_type):
        raise TypeError(f"Expected {expected_type}, but got {type(obj)}")

class TestRollupResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[419]  # Get the 420th JSON element (index 419)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the requested JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Test logic for ground truth 'rollup' function
                exec_globals = {
                    'pd': pd,
                    'PandasDataFrame': PandasDataFrame,
                    'PandasGroupedFrame': PandasGroupedFrame,
                    'Column': Column,
                    'Func': Func,
                    '_check_type': _check_type,
                }
                exec_locals = {}

                try:
                    # Dynamically execute code snippet
                    exec(code, exec_globals, exec_locals)

                    # Verify rollup function existence
                    if 'rollup' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'rollup' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "rollup",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    rollup = exec_locals['rollup']

                    # Test cases for rollup function
                    data = pd.DataFrame({
                        'A': [1, 2, 3, 4],
                        'B': [5, 6, 7, 8]
                    })

                    result_df = rollup(data, over={'A': ('A', 'sum'), 'B': ('B', 'sum')})
                    expected_rows = 1
                    expected_columns = 2

                    # Validate the result
                    self.assertEqual(result_df.shape, (expected_rows, expected_columns), 
                                     f"Code snippet {i} produced wrong DataFrame shape.")
                    
                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "rollup",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "rollup",
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

        # Remove old records related to "rollup"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "rollup"
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