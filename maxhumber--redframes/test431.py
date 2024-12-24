import unittest
import json
import os
import warnings
import pandas as pd
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestRankFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[430]  # Get the 431st JSON element
        if len(cls.code_snippet) < 1:
            raise ValueError("Expected at least one code snippet in the chosen JSON array")

    def test_rank_function(self):
        """Test the rank function with specific logic."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        code = self.code_snippet
        exec_globals = {}
        exec_locals = {}

        try:
            # Execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if rank is defined
            self.assertIn(
                'rank', exec_locals, "The 'rank' function is not found in the executed code."
            )

            rank = exec_locals['rank']

            # Define a test DataFrame
            df = pd.DataFrame({
                'A': [10, 20, 30, 20],
            })

            # Test 1: Basic ranking ascending
            result_df = rank(df, 'A', 'Rank')
            expected_ranks = [1.0, 2.0, 4.0, 2.0]
            pd.testing.assert_series_equal(
                result_df['Rank'],
                pd.Series(expected_ranks, name='Rank'),
                check_dtype=False
            )

            # Test 2: Basic ranking descending
            result_df_desc = rank(df, 'A', 'Rank', descending=True)
            expected_ranks_desc = [4.0, 2.5, 1.0, 2.5]
            pd.testing.assert_series_equal(
                result_df_desc['Rank'],
                pd.Series(expected_ranks_desc, name='Rank'),
                check_dtype=False
            )

            # Test 3: Check warning on overwriting existing column
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                rank(df, 'A', 'A')  # Attempt to overwrite existing column
                self.assertTrue(
                    any("overwriting existing column 'A'" in str(warn.message) for warn in w)
                )

            print("All assertions passed.\n")
            passed_count += 3  # Three tests passed
            results.append({"function_name": "rank", "result": "passed"})

        except Exception as e:
            print(f"Test failed with error: {e}\n")
            failed_count += 1
            results.append({"function_name": "rank", "code": code, "result": "failed"})

        # Final test summary
        print(f"Test Summary: {passed_count} passed, {failed_count} failed.\n")
        self.assertEqual(passed_count, 3, "Not all tests passed as expected!")

        # Write the results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for the function_name "rank"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "rank"
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