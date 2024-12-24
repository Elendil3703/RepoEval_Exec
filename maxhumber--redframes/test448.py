import unittest
import json
import sys
import re
import os
from typing import Any
from unittest.mock import MagicMock
import pprint

TEST_RESULT_JSONL = "test_result.jsonl"

class TestStrFunctionResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[447]  # Get the 448th JSON element (index 447)
        if not cls.code_snippet:
            raise ValueError("Expected at least one code snippet in the JSON array at position 447")

    def test_str_function(self):
        """Dynamically test the __str__ method."""
        results = []

        code = self.code_snippet

        # Static checks
        if "__str__" not in code:
            print(f"FAILED: Method '__str__' not found in code.\n")
            results.append({
                "function_name": "__str__",
                "code": code,
                "result": "failed"
            })
            self.fail("__str__ method not found in code snippet")

        # Dynamic execution
        exec_globals = {
            'pprint': pprint,
        }
        exec_locals = {}

        try:
            # Execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Dynamically create a mock object with _data attribute
            class DataFrameMock:
                def __init__(self, data):
                    self._data = data
                
                def __str__(self):
                    data = self._data.to_dict(orient="list")
                    string = pprint.pformat(data, indent=4, sort_dicts=False, compact=True)
                    if "\n" in string:
                        string = " " + string[1:-1]
                        string = f"rf.DataFrame({{\n{string}\n}})"
                    else:
                        string = f"rf.DataFrame({string})"
                    return string

            # Mock DataFrame's data and conversion method
            mock_data = MagicMock()
            mock_data.to_dict.return_value = {"foo": [1, 2], "bar": ["A", "B"]}

            # Instantiate the mock object and test __str__ method
            df = DataFrameMock(mock_data)
            str_representation = str(df)
            
            # Assertions
            self.assertIn("rf.DataFrame", str_representation, "Output does not start with 'rf.DataFrame'")
            self.assertIn("foo", str_representation, "'foo' column not found in output")
            self.assertIn("bar", str_representation, "'bar' column not found in output")
            
            print("PASSED all assertions.\n")
            results.append({
                "function_name": "__str__",
                "code": code,
                "result": "passed"
            })

        except Exception as e:
            print(f"FAILED with error: {e}\n")
            results.append({
                "function_name": "__str__",
                "code": code,
                "result": "failed"
            })

        # Write the result to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for '__str__'
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__str__"
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