import unittest
import json
import os
import pandas as pd
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCombineFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[421]  # Get the specific JSON element for index 421
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_combine_function_snippet(self):
        """Test the combine function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Check for required components in the code
                if "def combine" not in code:
                    print(f"Code snippet {i}: FAILED, function 'combine' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "combine",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    "pd": pd,
                    "PandasDataFrame": pd.DataFrame,
                    "Warnings": Any  # Infinite warning type
                }
                exec_locals = {}
                
                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Ensure combine is available
                    if 'combine' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'combine' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "combine",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    combine = exec_locals['combine']

                    # Define test inputs
                    df = pd.DataFrame({
                        'col1': ['A', 'B', 'C'],
                        'col2': ['D', 'E', 'F']
                    })
                    columns = ['col1', 'col2']
                    into = 'combined'
                    sep = '-'
                    
                    # Test the combine function
                    result_df = combine(df, columns, into, sep)

                    # Assertions
                    self.assertIn(into, result_df.columns, f"Code snippet {i} did not create the 'into' column.")
                    self.assertTrue(all(result_df[into] == ['A-D', 'B-E', 'C-F']), f"Code snippet {i} did not combine columns correctly.")

                    # Example verifying no columns dropped when 'drop' is False
                    result_df_no_drop = combine(df, columns, into, sep, drop=False)
                    self.assertTrue(all(col in result_df_no_drop.columns for col in columns), f"Code snippet {i} incorrectly dropped columns when drop=False.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "combine",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "combine",
                        "code": code,
                        "result": "failed"
                    })

        # Final test summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Writing to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        existing_records = [rec for rec in existing_records if rec.get("function_name") != "combine"]
        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()