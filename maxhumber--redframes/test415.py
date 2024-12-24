import unittest
import json
import os
import pandas as pd  # Ensure Pandas is imported
from typing import Any
from pandas import DataFrame as PandasDataFrame  # Alias for type hints

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGroundTruthFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        
        # Get the 415th JSON element (index 414)
        cls.code_snippets = data[414]
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 415th JSON array")

    def test_drop_function(self):
        """Dynamically test all code snippets for the 'drop' function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # ------------------- Static Check -------------------
                # Check globals definition
                if "def drop" not in code:
                    print(f"Code snippet {i}: FAILED, function 'drop' not defined.\n")
                    failed_count += 1
                    # Write failure record
                    results.append({
                        "function_name": "drop",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Pattern to check the drop function signature
                func_pattern = r"def\s+drop\s*\(.*?\)"
                if not re.search(func_pattern, code):
                    print(f"Code snippet {i}: FAILED, incorrect signature for 'drop'.\n")
                    failed_count += 1
                    # Write failure record
                    results.append({
                        "function_name": "drop",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic Execution -------------------
                exec_globals = {
                    'pd': pd,
                    'PandasDataFrame': PandasDataFrame,
                    'DataFrame': PandasDataFrame,  # For potential use in snippet
                    'Any': Any  # Inject Any if needed
                }
                exec_locals = {}

                try:
                    # Dynamically execute code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if 'drop' function exists
                    if 'drop' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'drop' function not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "drop",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    drop_function = exec_locals['drop']

                    # ------------------ Test Cases for 'drop' ------------------
                    # Test DataFrame
                    df = pd.DataFrame({
                        'A': [1, 2, 3],
                        'B': [4, 5, 6],
                        'C': [7, 8, 9]
                    })

                    # Test case 1: Drop a single column
                    result_df = drop_function(df, 'A')
                    expected_df = pd.DataFrame({
                        'B': [4, 5, 6],
                        'C': [7, 8, 9]
                    })
                    pd.testing.assert_frame_equal(result_df, expected_df)
                    
                    # Test case 2: Drop multiple columns
                    result_df = drop_function(df, ['A', 'C'])
                    expected_df = pd.DataFrame({
                        'B': [4, 5, 6]
                    })
                    pd.testing.assert_frame_equal(result_df, expected_df)

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "drop",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "drop",
                        "code": code,
                        "result": "failed"
                    })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Writing test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove existing records for function_name == "drop"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "drop"
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