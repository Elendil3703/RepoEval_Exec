import unittest
import json
import os
import pandas as pd
from typing import Any  # 确保注入的环境中有 Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCrossFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[426]  # Get the 427th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 427th JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect the test results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                # ------------------- Static Checks -------------------
                if "def cross" not in code:
                    print(f"Code snippet {i}: FAILED, function 'cross' not found.\n")
                    failed_count += 1
                    # Write failure record
                    results.append({
                        "function_name": "cross",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic Execution and Logic Testing -------------------
                exec_globals = {
                    'pd': pd,
                    'Any': Any  # Inject Any
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if 'cross' really exists
                    if 'cross' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'cross' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "cross",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Define test data
                    lhs_data = {'id': [1, 2]}
                    rhs_data = {'value': ['A', 'B']}
                    lhs_df = pd.DataFrame(lhs_data)
                    rhs_df = pd.DataFrame(rhs_data)

                    # Execute the tested function with test data
                    try:
                        result = exec_locals['cross'](lhs_df, rhs_df)
                        expected_data = {
                            'id_lhs': [1, 1, 2, 2],
                            'value_rhs': ['A', 'B', 'A', 'B']
                        }
                        expected_df = pd.DataFrame(expected_data)

                        pd.testing.assert_frame_equal(result, expected_df, check_like=True)
                        print(f"Code snippet {i}: PASSED all assertions.\n")
                        passed_count += 1
                        results.append({
                            "function_name": "cross",
                            "code": code,
                            "result": "passed"
                        })
                    except Exception as e:
                        print(f"Code snippet {i}: FAILED during execution with error: {e}\n")
                        failed_count += 1
                        results.append({
                            "function_name": "cross",
                            "code": code,
                            "result": "failed"
                        })

                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "cross",
                        "code": code,
                        "result": "failed"
                    })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Save test results to test_result.jsonl =============
        # Read existing test_result.jsonl (ignore if nonexistent)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "cross"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "cross"
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