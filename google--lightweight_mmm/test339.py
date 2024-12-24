import unittest
import json
import pandas as pd
import numpy as np
import jax.numpy as jnp
import os
from typing import List  # Ensure the necessary type is imported

TEST_RESULT_JSONL = "test_result.jsonl"

class TestSplitArrayIntoListResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the specific test code snippet
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[338]  # Get the 339th JSON element (index 338 in Python)

    def test_split_array_into_list(self):
        """Dynamically test the _split_array_into_list function."""
        results = []  # Collect results to write to JSONL

        # Define test scenario
        # Construct a sample dataframe
        sample_data = {
            'split_feature': ['A', 'A', 'B', 'B', 'C', 'C'],
            'feature1': [1, 2, 3, 4, 5, 6],
            'feature2': [7, 8, 9, 10, 11, 12]
        }
        dataframe = pd.DataFrame(sample_data)
        split_feature = 'split_feature'
        features = ['feature1', 'feature2']

        exec_globals = {
            'pd': pd,
            'np': np,
            'jnp': jnp,
        }
        exec_locals = {}

        try:
            # Dynamically execute the provided code snippet
            exec(self.code_snippet, exec_globals, exec_locals)

            # Check if the function exists in the executed locals
            if '_split_array_into_list' not in exec_locals:
                raise ValueError("'_split_array_into_list' not found in the executed code snippet.")

            # Retrieve the function
            split_func = exec_locals['_split_array_into_list']

            # Run test cases with different settings for national_model_flag
            result_with_flag = split_func(dataframe, split_feature, features, national_model_flag=True)
            result_without_flag = split_func(dataframe, split_feature, features, national_model_flag=False)

            # Validate results
            expected_shape_with_flag = (3, 2)  # Expected shape with national_model_flag=True
            expected_shape_without_flag = (3, 2, 1)  # Expected shape without squeezing

            self.assertEqual(result_with_flag.shape, expected_shape_with_flag, "Failed test with national_model_flag=True")
            self.assertEqual(result_without_flag.shape, expected_shape_without_flag, "Failed test with national_model_flag=False")

            print("Function _split_array_into_list: PASSED all assertions.\n")
            results.append({
                "function_name": "_split_array_into_list",
                "code": self.code_snippet,
                "result": "passed"
            })

        except Exception as e:
            print(f"Function _split_array_into_list: FAILED with error: {e}\n")
            results.append({
                "function_name": "_split_array_into_list",
                "code": self.code_snippet,
                "result": "failed"
            })

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        existing_records.append(json.loads(line.strip()))

        # Remove old records for "_split_array_into_list"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_split_array_into_list"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite the log file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()