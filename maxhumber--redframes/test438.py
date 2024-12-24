import unittest
import json
import os
import sys
from typing import Any
import pandas as pd
from pandas.testing import assert_frame_equal

TEST_RESULT_JSONL = "test_result.jsonl"

class TestTakeFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[437]  # Get the 438th JSON element (0-indexed)

    def test_take_function(self):
        """Test the 'take' function with various edge cases."""

        code = self.code_snippet
        exec_globals = {'Any': Any, 'PandasDataFrame': pd.DataFrame, 'PandasGroupedFrame': pd.DataFrame}

        try:
            # Dynamically execute the code snippet to define the take function
            exec(code, exec_globals)

            # Retrieve the defined 'take' function
            take = exec_globals['take']

            # Generate some test data
            df = pd.DataFrame({'A': range(10), 'B': range(10, 20)})

            # Expected results for various test cases
            expected_df_1 = df.head(1).reset_index(drop=True)
            expected_df_5 = df.head(5).reset_index(drop=True)
            expected_df_tail_3 = df.tail(3).reset_index(drop=True)

            # Test case: take 1 row
            result_df = take(df, 1)
            assert_frame_equal(result_df, expected_df_1, check_dtype=False)

            # Test case: take 5 rows
            result_df = take(df, 5)
            assert_frame_equal(result_df, expected_df_5, check_dtype=False)

            # Test case: take last 3 rows
            result_df = take(df, -3)
            assert_frame_equal(result_df, expected_df_tail_3, check_dtype=False)
            
            # Test case: take 0 rows (expect failure)
            with self.assertRaises(ValueError):
                take(df, 0)
            
            # Test case: take more rows than available (expect failure)
            with self.assertRaises(ValueError):
                take(df, 11)

            print("All take function test cases passed.\n")
            test_result = "passed"
        except Exception as e:
            print(f"Take function test failed with error: {e}\n")
            test_result = "failed"

        # Collect results
        result_record = {
            "function_name": "take",
            "code": code,
            "result": test_result
        }

        # Write result to JSONL file
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Delete function_name == "take" records
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "take"
        ]

        # Append new result
        existing_records.append(result_record)

        # Write to test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()