import unittest
import json
import sys
import os
import pandas as pd
import warnings
from typing import Any  # Ensure Any is available in the injected environment

TEST_RESULT_JSONL = "test_result.jsonl"

class TestAccumulateFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[425]  # Get the 426th JSON element (index 425)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 426th JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON with accumulate function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # ------------------- Logic checks -------------------
                # Static check: Verify if 'accumulate' function definition exists in the snippet
                if "def accumulate" not in code:
                    print(f"Code snippet {i}: FAILED, function 'accumulate' not found.\n")
                    failed_count += 1
                    # Record the failure
                    results.append({
                        "function_name": "accumulate",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic execution and testing -------------------
                # Prepare the environment for execution
                exec_globals = {
                    'pd': pd,
                    'warnings': warnings,
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if 'accumulate' function is present
                    if 'accumulate' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'accumulate' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "accumulate",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Tests for the accumulate function logic
                    accumulate = exec_locals['accumulate']

                    # Sample DataFrame
                    df = pd.DataFrame({
                        'values': [1, 2, 3, 4, 5],
                        'other_column': [10, 20, 30, 40, 50]
                    })

                    # Test 1: Accumulate on 'values' column
                    expected_result_1 = pd.Series([1, 3, 6, 10, 15], name='cumulative')
                    result_df_1 = accumulate(df, 'values', 'cumulative')
                    pd.testing.assert_series_equal(result_df_1['cumulative'], expected_result_1)
                    
                    # Test 2: Accumulate on 'other_column' with overwrite warning
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        expected_result_2 = pd.Series([10, 30, 60, 100, 150], name='values')
                        result_df_2 = accumulate(df, 'other_column', 'values')
                        pd.testing.assert_series_equal(result_df_2['values'], expected_result_2)

                        # Check if warning was raised
                        self.assertEqual(len(w), 1)
                        self.assertTrue(issubclass(w[-1].category, UserWarning))
                        self.assertIn("overwriting existing column 'values'", str(w[-1].message))

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "accumulate",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "accumulate",
                        "code": code,
                        "result": "failed"
                    })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write the test results to test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for function_name "accumulate"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "accumulate"
        ]

        # Append the new results
        existing_records.extend(results)

        # Rewrite the test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()