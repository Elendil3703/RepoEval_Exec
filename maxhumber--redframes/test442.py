import unittest
import json
import os
from typing import Any
import pandas as pd
from pandas import DataFrame as PandasDataFrame
import sys

TEST_RESULT_JSONL = "test_result.jsonl"


class TestLoadFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[441]  # Get the 442nd JSON element
        if len(cls.code_snippet) < 1:
            raise ValueError("Expected at least one code snippet in the 442nd JSON array")

    def test_load_function(self):
        """Dynamically test the load function in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect output results to write into JSONL

        code = self.code_snippet
        with self.subTest():
            try:
                print("Running dynamic test for load function...")

                # Attempt to execute the code snippet dynamically
                exec_globals = {
                    'pd': pd,
                    'sys': sys,
                    'DataFrame': PandasDataFrame,
                    'Any': Any,
                }
                exec(code, exec_globals)

                # Verify the load function is defined
                load_func = exec_globals.get('load', None)
                if load_func is None:
                    print("FAILED, 'load' function not found in exec_globals.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "load",
                        "code": code,
                        "result": "failed"
                    })
                    return

                # Define mock helper functions and return types to simulate _check_type, _check_file, etc.
                def _mock_check_type(path, expected_type):
                    assert isinstance(path, expected_type), "Mock type check failed."

                def _mock_check_file(path):
                    # Assume the file check passes
                    pass

                def _mock_check_index(data):
                    # Assume the index check passes
                    pass

                def _mock_check_columns(data):
                    # Assume the columns check passes
                    pass

                def _mock_wrap(data):
                    # Simulate wrapping data
                    return data

                # Replace internal functions with mock functions during the test
                exec_globals['_check_type'] = _mock_check_type
                exec_globals['_check_file'] = _mock_check_file
                exec_globals['_check_index'] = _mock_check_index
                exec_globals['_check_columns'] = _mock_check_columns
                exec_globals['_wrap'] = _mock_wrap

                # Test the load function with a mock file path
                mock_csv_path = "mock_example.csv"
                expected_result = PandasDataFrame()  # Empty DataFrame as a mock result

                # Emulate pd.read_csv behavior with a mock
                def _mock_read_csv(path, **kwargs):
                    assert path == mock_csv_path, "Unexpected file path."
                    return expected_result

                exec_globals['pd.read_csv'] = _mock_read_csv

                # Invoke the load function
                result = load_func(mock_csv_path)
                self.assertIsInstance(result, PandasDataFrame, "The result should be a DataFrame")
                self.assertEqual(result, expected_result, "The result should match the expected DataFrame")

                print("PASSED all assertions.\n")
                passed_count += 1
                results.append({
                    "function_name": "load",
                    "code": code,
                    "result": "passed"
                })
            except Exception as e:
                print(f"FAILED with error: {e}\n")
                failed_count += 1
                results.append({
                    "function_name": "load",
                    "code": code,
                    "result": "failed"
                })

        # Print summary
        print(f"Test Summary: {passed_count} passed, {failed_count} failed, total 1\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # Write the results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        existing_records.append(json.loads(line))

        # Remove old records for function_name == "load"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "load"
        ]

        # Append new results
        existing_records.extend(results)

        # Write back to JSONL
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    unittest.main()