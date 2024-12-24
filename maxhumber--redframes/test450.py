import unittest
import json
import os
from typing import Any
import pandas as pd
import numpy as np

TEST_RESULT_JSONL = "test_result.jsonl"

class TestColumnTypes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[449]  # Get the 450th JSON element (index 449)
        if not cls.code_snippet:
            raise ValueError("Expected at least one code snippet in the specified JSON element")

    def test_column_types(self):
        """Dynamically test the types method in the JSON code."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Used to collect test results to write to JSONL

        code = self.code_snippet

        exec_globals = {
            'np': np,
            'pd': pd,
            'Any': Any
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Extract the DataFrame class with the types method
            DataFrame = exec_locals.get('DataFrame')

            if DataFrame is None:
                results.append({
                    "function_name": "types",
                    "code": code,
                    "result": "failed"
                })
                raise AssertionError("DataFrame class not found in code snippet.")

            # Create a sample DataFrame and test the types method
            df = DataFrame({
                "foo": [1, 2],
                "bar": ["A", "B"],
                "baz": [True, False]
            })
            
            expected_types = {
                'foo': int,
                'bar': object,
                'baz': bool
            }

            # Check types method output
            self.assertEqual(df.types(), expected_types, "Column types do not match expected types.")
          
            print("Code snippet: PASSED all assertions.")
            passed_count += 1
            results.append({
                "function_name": "types",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}")
            failed_count += 1
            results.append({
                "function_name": "types",
                "code": code,
                "result": "failed"
            })

        # Results summary
        print(f"Test Summary: {passed_count} passed, {failed_count} failed, total 1\n")

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old "types" function records
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "types"
        ]

        # Append new results
        existing_records.extend(results)

        # Write the results back
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")


if __name__ == "__main__":
    unittest.main()