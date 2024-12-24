import unittest
import json
import os
import pandas as pd
import uuid
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

def _check_type(value, _type):
    if not isinstance(value, _type):
        raise TypeError(f"Expected type {_type} for {value}, but got {type(value)}")

class TestSplitFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON data
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[416]  # Get the specific request at index 416
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 417th JSON array")

    def test_split_function(self):
        """Test the split function from the provided code snippets."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                if "def split" not in code:
                    print(f"Code snippet {i}: FAILED, 'split' function not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "split",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {'pd': pd, 'uuid': uuid, '_check_type': _check_type}
                exec_locals = {}

                try:
                    exec(code, exec_globals, exec_locals)

                    if 'split' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'split' function not in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "split",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    split = exec_locals['split']

                    # Test case 1
                    df = pd.DataFrame({'full_name': ['John Doe', 'Jane Smith']})
                    expected_df = pd.DataFrame({'first_name': ['John', 'Jane'], 'last_name': ['Doe', 'Smith']})
                    result_df = split(df, 'full_name', ['first_name', 'last_name'], ' ', drop=True)
                    pd.testing.assert_frame_equal(result_df, expected_df)

                    # Test case 2: Non-unique 'into' keys
                    try:
                        split(df, 'full_name', ['first_name', 'first_name'], ' ')
                        failed_count += 1
                        results.append({
                            "function_name": "split",
                            "code": code,
                            "result": "failed"
                        })
                        print(f"Code snippet {i}: FAILED, no error for non-unique 'into' keys.\n")
                        continue
                    except KeyError:
                        pass  # Expected behavior

                    # Test case 3: `drop` is False and column name in `into`
                    try:
                        split(df, 'full_name', ['first_name', 'full_name'], ' ', drop=False)
                        failed_count += 1
                        results.append({
                            "function_name": "split",
                            "code": code,
                            "result": "failed"
                        })
                        print(f"Code snippet {i}: FAILED, no error for column name in 'into' when drop is False.\n")
                        continue
                    except KeyError:
                        pass  # Expected behavior

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "split",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "split",
                        "code": code,
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write the results to the JSONL file
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        existing_records = [rec for rec in existing_records if rec.get("function_name") != "split"]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()