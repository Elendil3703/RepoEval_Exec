import unittest
import json
import os
import pandas as pd
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestSampleFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[427]  # Get the 428th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 428th JSON array")

    def test_sample_function_snippets(self):
        """Dynamically test all code snippets regarding the sample function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # ------------------- Static checks -------------------
                if "def sample" not in code:
                    print(f"Code snippet {i}: FAILED, function 'sample' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "sample",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic execution and tests -------------------
                exec_globals = {
                    'pd': pd,
                    'Any': Any,  # Inject any additional globals if needed
                }
                exec_locals = {}

                try:
                    # Dynamically execute code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if 'sample' is defined in the executed local scope
                    if 'sample' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'sample' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "sample",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    sample_function = exec_locals['sample']

                    # Test the sample function with different scenarios
                    df = pd.DataFrame({'A': range(10)})

                    # Test with integer rows
                    sample_result = sample_function(df, 5, seed=42)
                    self.assertEqual(len(sample_result), 5, f"Code snippet {i} failed on integer sampling.")

                    # Test with fraction rows
                    sample_result = sample_function(df, 0.5, seed=42)
                    self.assertEqual(len(sample_result), 5, f"Code snippet {i} failed on fractional sampling.")

                    # Test with invalid input
                    with self.assertRaises(ValueError, msg=f"Code snippet {i} failed to raise ValueError on >1 float."):
                        sample_function(df, 1.5)

                    with self.assertRaises(ValueError, msg=f"Code snippet {i} failed to raise ValueError on <=0."):
                        sample_function(df, 0)

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "sample",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "sample",
                        "code": code,
                        "result": "failed"
                    })

        # Test summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for 'sample'
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "sample"
        ]

        # Append new results
        existing_records.extend(results)

        # Re-write test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()