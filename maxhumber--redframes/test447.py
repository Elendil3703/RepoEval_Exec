import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class DataFrame:
    def __init__(self, data):
        self._data = data

    def equals(self, other):
        """Simulates Pandas DataFrame.equals()"""
        return self._data == other._data

class TestDataFrameEquality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.test_case = data[446]  # Get the 447th JSON element (index 446)

    def test_dataframe_equality(self):
        """Dynamically test the __eq__ implementation."""
        results = []

        # Parse code snippet
        code = self.test_case

        exec_globals = {
            'Any': Any,
            'DataFrame': DataFrame,
        }
        exec_locals = {}

        try:
            # Execute the code to define the __eq__ method
            exec(code, exec_globals, exec_locals)

            # Create DataFrame instances for testing
            df1 = DataFrame({"foo": [1]})
            df2 = DataFrame({"bar": [1]})
            df3 = DataFrame({"foo": [1]})

            # Test equality
            self.assertFalse(df1 == df2, "df1 should not equal df2.")
            self.assertTrue(df1 == df3, "df1 should equal df3.")
            self.assertFalse(df1 == {"foo": [1]}, "df1 should not equal a non-DataFrame object.")

            results.append({
                "function_name": "__eq__",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"Test failed with error: {e}")
            results.append({
                "function_name": "__eq__",
                "code": code,
                "result": "failed"
            })

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for __eq__
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__eq__"
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