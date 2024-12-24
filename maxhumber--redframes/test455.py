import unittest
import json
import os
import re
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCheckColumnsResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[454]  # Get the 455th JSON element (index 454)
        if not cls.code_snippet:
            raise ValueError("Expected a code snippet.")

    def test_check_columns(self):
        """Dynamically test the _check_columns function in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write into JSONL

        code = self.code_snippet
        with self.subTest(code=code):
            print("Running test for the code snippet...")
            # Static check: verify that _check_columns is defined
            if "_check_columns" not in code:
                print("Code snippet: FAILED, '_check_columns' not found in code.\n")
                failed_count += 1
                results.append({
                    "function_name": "_check_columns",
                    "code": code,
                    "result": "failed"
                })
                return

            # Dynamic execution and logic testing
            exec_globals = {
                'PandasDataFrame': type('PandasDataFrameMock', (), {}),
                'PandasIndex': type('PandasIndexMock', (), {}),
                'Any': Any,
            }
            exec_locals = {}

            try:
                # Dynamically execute the code snippet
                exec(code, exec_globals, exec_locals)

                # Additional dynamic checks
                df_mock = type('MockDataFrame', (), {
                    'columns': type('MockIndex', (), {'has_duplicates': False})
                })

                # Pass the dynamic test checking no duplicate columns
                try:
                    exec_locals["_check_columns"](df_mock())
                    print("No error raised for valid columns, as expected.")
                except Exception as e:
                    print(f"Code snippet: FAILED with unexpected error: {e}\n")
                    raise

                # Test: simulate having duplicate columns
                df_mock_with_duplicates = type('MockDataFrame', (), {
                    'columns': type('MockIndex', (), {'has_duplicates': True})
                })
                try:
                    exec_locals["_check_columns"](df_mock_with_duplicates())
                    print("Code snippet: FAILED, expected KeyError for duplicate columns.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_check_columns",
                        "code": code,
                        "result": "failed"
                    })
                except KeyError as e:
                    print("Duplicate columns correctly raised KeyError as expected.")
                    passed_count += 1
                    results.append({
                        "function_name": "_check_columns",
                        "code": code,
                        "result": "passed"
                    })

            except Exception as e:
                print(f"Code snippet: FAILED with error: {e}\n")
                failed_count += 1
                results.append({
                    "function_name": "_check_columns",
                    "code": code,
                    "result": "failed"
                })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 1\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # Write the test results into test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for _check_columns
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_check_columns"
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