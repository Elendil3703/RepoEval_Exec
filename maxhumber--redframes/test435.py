import unittest
import json
import os
from typing import Any
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy as PandasGroupedFrame
from pandas import DataFrame as PandasDataFrame

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGroupedMelt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[434]  # Get the 435th JSON element (index 434)
        if not cls.code_snippet:
            raise ValueError("Expected code snippet at index 434 in the JSON data.")

    def test_grouped_melt(self):
        """Test the _grouped_melt function with sample data."""
        results = []
        
        code = self.code_snippet
        exec_globals = {
            'pd': pd,
            'PandasGroupedFrame': PandasGroupedFrame,
            'PandasDataFrame': PandasDataFrame,
            'Any': Any
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet to define _grouped_melt
            exec(code, exec_globals, exec_locals)

            if '_grouped_melt' not in exec_locals:
                print("_grouped_melt function is not defined in the code snippet.")
                results.append({
                    "function_name": "_grouped_melt",
                    "code": code,
                    "result": "failed"
                })
                raise ValueError("_grouped_melt function is not defined.")

            # Sample data for testing
            data = {'category': ['A', 'A', 'B', 'B'],
                    'type': ['X', 'Y', 'X', 'Y'],
                    'value1': [1, 2, 3, 4],
                    'value2': [5, 6, 7, 8]}
            df = pd.DataFrame(data)
            grouped = df.groupby(['category'])

            # Test the _grouped_melt function
            result_df = exec_locals['_grouped_melt'](grouped, ('variable', 'value'))
            
            # Define expected result
            expected_data = {
                'category': ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B'],
                'variable': ['value1', 'value1', 'value1', 'value1', 'value2', 'value2', 'value2', 'value2'],
                'value': [1, 2, 3, 4, 5, 6, 7, 8]
            }
            expected_df = pd.DataFrame(expected_data)

            # Assert the result is as expected
            pd.testing.assert_frame_equal(result_df.reset_index(drop=True), expected_df)

            print("Test passed for '_grouped_melt'.")
            results.append({
                "function_name": "_grouped_melt",
                "code": code,
                "result": "passed"
            })

        except Exception as e:
            print(f"Test failed for '_grouped_melt' with error: {e}")
            results.append({
                "function_name": "_grouped_melt",
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

        # Remove old records for _grouped_melt
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_grouped_melt"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()