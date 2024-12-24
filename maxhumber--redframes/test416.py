import unittest
import json
import os
import pandas as pd
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestRenameFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[415]  # Get the 416th JSON element

    def test_rename(self):
        """Test the `rename` function with various cases."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results for writing to JSONL

        if "rename" not in self.code_snippet:
            print(f"Code snippet does not contain 'rename' function.")
            failed_count += 1
            results.append({
                "function_name": "rename",
                "code": self.code_snippet,
                "result": "failed"
            })
            return

        exec_globals = {
            'pd': pd,
            'Any': Any
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(self.code_snippet, exec_globals, exec_locals)

            # Check if the function 'rename' exists
            self.assertIn('rename', exec_locals, "The 'rename' function is not defined.")

            # Import and define the function to test
            rename = exec_locals['rename']

            # Create a mock DataFrame
            df = pd.DataFrame({
                'A': [1, 2, 3],
                'B': [4, 5, 6]
            })

            # Test cases for the `rename` function
            test_cases = [
                [{'A': 'Alpha', 'B': 'Beta'}, {'Alpha', 'Beta'}, True],
                [{'A': 'Alpha', 'A': 'Alpha1'}, None, False],  # Duplicates should fail
                [{'X': 'Alpha'}, None, False],  # Missing keys
                [{'A': 'Alpha', 'X': 'Beta'}, None, False]  # Partial missing keys
            ]

            for i, (columns, expected_columns, should_pass) in enumerate(test_cases):
                with self.subTest(test_index=i):
                    try:
                        new_df = rename(df, columns)
                        if should_pass:
                            self.assertEqual(set(new_df.columns), expected_columns)
                        else:
                            self.fail(f"Test case {i} should have failed, but passed.")
                    except KeyError as e:
                        if should_pass:
                            self.fail(f"Test case {i} raised an unexpected error: {e}")
                        else:
                            pass
            print(f"All test cases passed successfully.")
            passed_count += len(test_cases)
            results.append({
                "function_name": "rename",
                "code": self.code_snippet,
                "result": "passed"
            })
        except Exception as e:
            print(f"Execution failed with error: {e}")
            failed_count += 1
            results.append({
                "function_name": "rename",
                "code": self.code_snippet,
                "result": "failed"
            })

        # ============= Write test results to test_result.jsonl =============
        # Read existing test_result.jsonl (if does not exist, ignore)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "rename"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "rename"
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