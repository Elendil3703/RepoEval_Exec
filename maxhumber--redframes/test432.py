import unittest
import json
import os
import pandas as pd
from typing import Any  # 确保注入的环境中有 Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestFillFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[431]  # Get the 432nd JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the provided JSON array")

    def test_fill_function(self):
        """Dynamically test fill functions in the JSON."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # 收集要写入 JSONL 的测试结果

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Ensure necessary components are present
                if "def fill" not in code:
                    print(f"Code snippet {i}: FAILED, function 'fill' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "fill",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'pd': pd,
                    'Any': Any,
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check fill function existence
                    if 'fill' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'fill' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "fill",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Prepare test data
                    df = pd.DataFrame({
                        'A': [1, None, 3],
                        'B': [None, 2, None]
                    })

                    # Execute fill function with the test data
                    filled_df = exec_locals['fill'](df, direction="down")

                    # Validations
                    expected_df = pd.DataFrame({
                        'A': [1, 1, 3],
                        'B': [None, 2, 2]
                    })

                    pd.testing.assert_frame_equal(filled_df, expected_df, check_dtype=False)

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "fill",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "fill",
                        "code": code,
                        "result": "failed"
                    })

        # Final test summary
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

        # Delete old "fill" function records
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "fill"
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