import unittest
import json
import os
from typing import Any, Callable
import pandas as pd

TEST_RESULT_JSONL = "test_result.jsonl"

class PandasDataFrame(pd.DataFrame):
    """A placeholder class to represent the Pandas DataFrame type."""
    pass

class Func(Callable):
    """A placeholder for the Func type, representing a callable."""
    pass

class TestFilterFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[436]  # Get the 437th JSON element
        if len(cls.code_snippet) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_filter_function(self):
        """Dynamically test the filter function with various conditions."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Gather test results for JSONL

        code = self.code_snippet

        # Prepare the execution environment
        exec_globals = {
            'Any': Any,
            'PandasDataFrame': PandasDataFrame,
            'Func': Func,
            'pd': pd,
        }
        exec_locals = {}

        try:
            # Execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Assert the presence of the 'filter' function
            self.assertIn('filter', exec_locals, "Function 'filter' was not defined in the code snippet.")

            # Reference to the 'filter' function
            filter_func = exec_locals['filter']

            # Test cases
            df = PandasDataFrame({
                'A': [1, 2, 3, 4],
                'B': [5, 6, 7, 8]
            })
            
            # Case 1: Basic filtering
            result = filter_func(df, lambda x: x['A'] > 2)
            self.assertEqual(len(result), 2, "Filtering did not produce the expected number of rows.")
            self.assertListEqual(result['A'].tolist(), [3, 4], "Filtered data does not match expected output.")

            passed_count += 1
            results.append({
                "function_name": "filter",
                "code": code,
                "result": "passed"
            })
        
        except Exception as e:
            print(f"Code snippet failed with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "filter",
                "code": code,
                "result": "failed"
            })

        # Summary of the results
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 1\n")
        
        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for 'filter'
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "filter"
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