import unittest
import json
import os
import pandas as pd
from pandas.testing import assert_frame_equal

TEST_RESULT_JSONL = "test_result.jsonl"

class TestSortFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON data
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[417]
        if not cls.code_snippet:
            raise ValueError("Expected code snippet in the JSON data")

    def test_sort_function(self):
        """Test the sort function with various dataframe scenarios."""
        passed_count = 0
        failed_count = 0
        results = []

        # Simulate the environment for the sort function
        exec_globals = {
            'pd': pd,
            'assert_frame_equal': assert_frame_equal,
        }
        exec_locals = {}

        try:
            # Execute the code snippet to define the sort function
            exec(self.code_snippet, exec_globals, exec_locals)

            # Get the sort function
            if 'sort' not in exec_locals:
                raise ValueError("sort function not defined in the code snippet")

            sort_func = exec_locals['sort']

            # Define test cases
            test_cases = [
                (
                    pd.DataFrame({'a': [3, 1, 2], 'b': [9, 8, 7]}),
                    {'columns': 'a', 'descending': False},
                    pd.DataFrame({'a': [1, 2, 3], 'b': [8, 7, 9]}),
                ),
                (
                    pd.DataFrame({'a': [3, 1, 2], 'b': [9, 8, 7]}),
                    {'columns': 'a', 'descending': True},
                    pd.DataFrame({'a': [3, 2, 1], 'b': [9, 7, 8]}),
                ),
                # Add more cases as needed
            ]

            for idx, (df, kwargs, expected) in enumerate(test_cases):
                with self.subTest(test_case_index=idx):
                    result = sort_func(df, **kwargs)
                    try:
                        assert_frame_equal(result, expected)
                        print(f"Test case {idx}: PASSED\n")
                        passed_count += 1
                        results.append({
                            "function_name": "sort",
                            "test_case_index": idx,
                            "result": "passed"
                        })

                    except AssertionError:
                        print(f"Test case {idx}: FAILED\n")
                        failed_count += 1
                        results.append({
                            "function_name": "sort",
                            "test_case_index": idx,
                            "result": "failed"
                        })

        except Exception as e:
            print(f"Code snippet execution failed with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "sort",
                "error": str(e),
                "result": "failed"
            })

        # Test summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed\n")
        self.assertEqual(passed_count + failed_count, len(test_cases),
                         "Test count mismatch with cases executed!")

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records related to sort
        existing_records = [
            rec for rec in existing_records if rec.get("function_name") != "sort"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite the test_result.jsonl file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()