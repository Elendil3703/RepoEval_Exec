import unittest
import json
import os
import pandas as pd
from typing import Tuple, Union  # Ensure the correct types are available

TEST_RESULT_JSONL = "test_result.jsonl"

class PandasDataFrame(pd.DataFrame):
    pass

class LazyColumns(list):
    pass

def _check_type(value, expected_types):
    if not isinstance(value, expected_types):
        raise TypeError(f"Expected type {expected_types}, but got {type(value)}.")

class TestJoinFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[435]  # Get the 436th JSON element

    def test_join(self):
        """Test the join function with various scenarios."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        # Prepare test data
        data1 = {'key': [1, 2, 3], 'value1': ['A', 'B', 'C']}
        data2 = {'key': [2, 3, 4], 'value2': ['D', 'E', 'F']}
        lhs = PandasDataFrame(data1)
        rhs = PandasDataFrame(data2)

        exec_globals = {"PandasDataFrame": PandasDataFrame, "pd": pd, "_check_type": _check_type, "LazyColumns": LazyColumns}
        exec_locals = {}

        try:
            # Dynamically execute the join function code snippet
            exec(self.code_snippet, exec_globals, exec_locals)

            if 'join' not in exec_locals:
                raise AssertionError("Function 'join' not found in executed locals.")

            # Test the join function with various join types
            join = exec_locals['join']

            # Test cases
            test_cases = [
                ("left", 3, ['A', 'B', 'C'], ['D', 'E']),
                ("right", 3, ['B', 'C'], ['D', 'E', 'F']),
                ("inner", 2, ['B', 'C'], ['D', 'E']),
                ("full", 4, ['A', 'B', 'C', None], [None, 'D', 'E', 'F']),
            ]

            for join_type, expected_length, lhs_values, rhs_values in test_cases:
                with self.subTest(join_type=join_type):
                    result = join(lhs, rhs, on='key', how=join_type)
                    self.assertEqual(len(result), expected_length, f"Failed on join type: {join_type}")
                    self.assertListEqual(list(result['value1'].dropna()), lhs_values, f"Failed on join type: {join_type} - in 'value1'")
                    self.assertListEqual(list(result['value2'].dropna()), rhs_values, f"Failed on join type: {join_type} - in 'value2'")
                    passed_count += 1
                    results.append({
                        "function_name": "join",
                        "how": join_type,
                        "code": self.code_snippet,
                        "result": "passed"
                    })

        except Exception as e:
            failed_count += 1
            results.append({
                "function_name": "join",
                "code": self.code_snippet,
                "result": "failed"
            })

        # Summary statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed\n")
        self.assertEqual(passed_count + failed_count, len(test_cases), "Test count mismatch!")

        # Write test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records where function_name == "join"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "join"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()