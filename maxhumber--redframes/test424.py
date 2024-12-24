import unittest
import json
import os
from typing import Any
import pandas as pd

TEST_RESULT_JSONL = "test_result.jsonl"

class TestDenixFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[423]  # Get the 424th JSON element (index 423)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 424th JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # ------------------- 多一些逻辑检查 -------------------
                if "def denix" not in code:
                    print(f"Code snippet {i}: FAILED, function 'denix' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "denix",
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
                    exec(code, exec_globals, exec_locals)

                    if 'denix' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'denix' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "denix",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    denix = exec_locals['denix']

                    # Test cases
                    df = pd.DataFrame({
                        'A': [1, 2, 3, None],
                        'B': [4, None, 6, 7]
                    })

                    # Case 1: Remove rows with NaNs in column 'A'
                    result = denix(df, 'A')
                    expected = pd.DataFrame({'A': [1, 2, 3], 'B': [4, None, 6]}).reset_index(drop=True)
                    pd.testing.assert_frame_equal(result, expected)

                    # Case 2: Remove rows with NaNs in columns ['A', 'B']
                    result = denix(df, ['A', 'B'])
                    expected = pd.DataFrame({'A': [1], 'B': [4]}).reset_index(drop=True)
                    pd.testing.assert_frame_equal(result, expected)

                    # Case 3: Pass None as columns, should return df unchanged
                    result = denix(df, None)
                    expected = df.dropna(subset=None).reset_index(drop=True)
                    pd.testing.assert_frame_equal(result, expected)

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "denix",
                        "code": code,
                        "result": "passed"
                    })

                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "denix",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary and JSONL writing
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "denix"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()