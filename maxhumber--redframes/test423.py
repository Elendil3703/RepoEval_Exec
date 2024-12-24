import unittest
import json
import os
import sys
from typing import Any
import pandas as pd

TEST_RESULT_JSONL = "test_result.jsonl"

class TestReplaceFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[422]  # Get the 423rd JSON element (index 422)
        if not cls.code_snippet:
            raise ValueError("Expected code snippet in the 423rd JSON element")

    def test_replace_function(self):
        """Dynamically test the replace function in the JSON with additional checks."""
        passed_count = 0
        failed_count = 0
        results = []

        code = self.code_snippet

        # Initialize globals for exec
        exec_globals = {
            'pd': pd,
            'Any': Any,
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if the function 'replace' was defined
            if 'replace' not in exec_locals:
                print("FAILED, 'replace' function not found in exec_locals.\n")
                failed_count += 1
                results.append({
                    "function_name": "replace",
                    "code": code,
                    "result": "failed"
                })
            else:
                # Test the replace function
                replace_fn = exec_locals['replace']

                # Create test data
                df = pd.DataFrame({
                    'A': [1, 2, 3],
                    'B': ['x', 'y', 'z']
                })

                # Test case 1: Valid replacement
                over = {'A': {1: 10, 2: 20}}
                result_df = replace_fn(df, over)
                expected_df = pd.DataFrame({
                    'A': [10, 20, 3],
                    'B': ['x', 'y', 'z']
                })
                pd.testing.assert_frame_equal(result_df, expected_df)
                
                # Test case 2: Invalid column
                try:
                    replace_fn(df, {'C': {1: 10}})
                    print("FAILED, expected KeyError for invalid column 'C'.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "replace",
                        "code": code,
                        "result": "failed"
                    })
                    return
                except KeyError as e:
                    assert "column key: ['C'] is invalid" in str(e)
                
                # Test case 3: Invalid multiple columns
                try:
                    replace_fn(df, {'A': {1: 10}, 'C': {1: 10}})
                    print("FAILED, expected KeyError for invalid columns 'C'.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "replace",
                        "code": code,
                        "result": "failed"
                    })
                    return
                except KeyError as e:
                    assert "column keys: ['C'] are invalid" in str(e)

                # If all assertions pass
                print("PASSED all assertions.\n")
                passed_count += 1
                results.append({
                    "function_name": "replace",
                    "code": code,
                    "result": "passed"
                })
        except Exception as e:
            print(f"FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "replace",
                "code": code,
                "result": "failed"
            })

        # Test summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # Write to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for the function 'replace'
        existing_records = [
            record for record in existing_records
            if record.get("function_name") != "replace"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()