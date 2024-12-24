import unittest
import json
import sys
import re
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestSetupMockMMMResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[326]  # Get the 327th JSON element (index 326)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 327th JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # Static check: Ensure function 'set_up_mock_mmm' is defined
                if "def set_up_mock_mmm" not in code:
                    print(f"Code snippet {i}: FAILED, function 'set_up_mock_mmm' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "set_up_mock_mmm",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Static check: Ensure 'LightweightMMM' is addressed in the code
                if "LightweightMMM" not in code:
                    print(f"Code snippet {i}: FAILED, 'LightweightMMM' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "set_up_mock_mmm",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Dynamic execution and logic test
                exec_globals = {
                    'sys': sys,
                    'Any': Any,
                }
                exec_locals = {}

                try:
                    # Attempt to execute the code
                    exec(code, exec_globals, exec_locals)

                    # Test the 'set_up_mock_mmm' function with mock inputs
                    model_name = "adstock"
                    is_geo_model = True

                    mmm = exec_locals['set_up_mock_mmm'](model_name, is_geo_model)
                    
                    # Check if the function returned a correct LightweightMMM object
                    self.assertEqual(mmm.n_media_channels, 5, f"Code snippet {i} returned incorrect n_media_channels.")
                    self.assertEqual(mmm.n_geos, 3, f"Code snippet {i} returned incorrect n_geos for geo model.")
                    self.assertEqual(len(mmm.media_names), 5, f"Code snippet {i} returned incorrect media names.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "set_up_mock_mmm",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "set_up_mock_mmm",
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

        # Remove old records for function_name == "set_up_mock_mmm"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "set_up_mock_mmm"
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