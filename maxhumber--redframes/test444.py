import unittest
import json
import os
from typing import Any
from pandas import DataFrame as PandasDataFrame
from pandas.core.frame import DataFrame

TEST_RESULT_JSONL = "test_result.jsonl"

class TestWrapFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[443]  # Get the 444th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 444th JSON array")

    def test_wrap_function(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results for JSONL writing

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Check for function '_wrap' signature presence
                if "def _wrap" not in code:
                    print(f"Code snippet {i}: FAILED, function '_wrap' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_wrap",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Execute the code snippet
                exec_globals = {
                    'PandasDataFrame': PandasDataFrame,
                    'DataFrame': DataFrame,
                }
                exec_locals = {}

                try:
                    # Dynamically execute code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if '_wrap' function exists
                    if '_wrap' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, '_wrap' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "_wrap",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Prepare a test DataFrame
                    original_data = PandasDataFrame({'a': [1, 2], 'b': [3, 4]})

                    # Call _wrap
                    wrapped_df = exec_locals['_wrap'](original_data)

                    # Assertions
                    self.assertTrue(
                        isinstance(wrapped_df, DataFrame),
                        f"Code snippet {i}: The returned object is not a DataFrame."
                    )
                    self.assertIs(
                        wrapped_df._data, 
                        original_data,
                        f"Code snippet {i}: The wrapped data is not the same as the original data."
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "_wrap",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_wrap",
                        "code": code,
                        "result": "failed"
                    })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write test results to test_result.jsonl
        # Read existing test_result.jsonl (if any)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for '_wrap'
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_wrap"
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