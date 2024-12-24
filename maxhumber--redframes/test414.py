import unittest
import json
import sys
import os
from typing import Any, Dict, Callable
import pandas as pd

TEST_RESULT_JSONL = "test_result.jsonl"

class TestMutateFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[413]  # Get the 414th JSON element (index 413)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 414th JSON array")

    def test_code_snippets(self):
        """Test all code snippets in the JSON with specific checks for 'mutate'."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # ------------------- Static code checks -------------------
                # Check for 'mutate' function signature
                if "def mutate" not in code:
                    print(f"Code snippet {i}: FAILED, function 'mutate' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "mutate",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic execution and logical test -------------------
                exec_globals = {
                    'Any': Any, 
                    'Dict': Dict, 
                    'Callable': Callable,
                    'pd': pd
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)
                    
                    # Check if 'mutate' is defined
                    if 'mutate' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'mutate' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "mutate",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test mutate function
                    mutate = exec_locals['mutate']

                    # Create test DataFrame
                    df = pd.DataFrame({
                        "A": [1, 2, 3],
                        "B": [4, 5, 6]
                    })

                    # Define mutations
                    mutations = {
                        "A": lambda row: row["A"] + 10,
                        "B": lambda row: row["B"] * 2
                    }

                    # Expected result after mutation
                    expected_df = pd.DataFrame({
                        "A": [11, 12, 13],
                        "B": [8, 10, 12]
                    })

                    # Call the function and test
                    result_df = mutate(df, mutations)
                    
                    # Assert the result matches expected
                    pd.testing.assert_frame_equal(result_df, expected_df)

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "mutate",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "mutate",
                        "code": code,
                        "result": "failed"
                    })

        # Summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for 'mutate'
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "mutate"
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