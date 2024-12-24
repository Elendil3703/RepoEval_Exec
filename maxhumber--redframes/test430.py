import unittest
import json
import sys
import os
import pandas as pd
from typing import Any  # Ensure Any is available in the injected environment

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCarperAIUnpackResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[429]  # Get the 430th JSON element (index 429)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON for the 'unpack' function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write into the JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # Verify that `unpack` is defined in the code snippet
                if "def unpack" not in code:
                    print(f"Code snippet {i}: FAILED, 'unpack' function not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "unpack",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Prepare a mock dataframe for testing the `unpack` function
                mock_data = {'col1': ['a,b,c', 'd,e', 'f']}
                df = pd.DataFrame(mock_data)

                exec_globals = {
                    'pd': pd,
                    '_check_type': lambda x, y: isinstance(x, y),  # Mock function
                    'Any': Any,
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if `unpack` really exists
                    if 'unpack' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'unpack' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "unpack",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Call the `unpack` function
                    unpack_func = exec_locals['unpack']
                    result_df = unpack_func(df, 'col1', ',')

                    # Test that the result is a DataFrame with the expected results
                    expected_data = {'col1': ['a', 'b', 'c', 'd', 'e', 'f']}
                    expected_df = pd.DataFrame(expected_data)
                    
                    pd.testing.assert_frame_equal(result_df, expected_df)

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "unpack",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "unpack",
                        "code": code,
                        "result": "failed"
                    })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Writing test results to test_result.jsonl
        # Read existing test_result.jsonl (ignore if not exist)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records where function_name == "unpack"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "unpack"
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