import unittest
import json
import sys
import re
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCarperAITrlxResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[324]  # Get the 325th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 325th JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # ------------------- Static Checks -------------------
                # Check the presence of specific keys and function definitions.
                if "immutabledict" not in code:
                    print(f"Code snippet {i}: FAILED, 'immutabledict' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_get_transform_default_priors",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                if "def _get_transform_default_priors" not in code:
                    print(f"Code snippet {i}: FAILED, function '_get_transform_default_priors' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_get_transform_default_priors",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Function signature regex check
                func_pattern = r"def\s+_get_transform_default_priors\s*\("
                if not re.search(func_pattern, code):
                    print(f"Code snippet {i}: FAILED, incorrect signature for '_get_transform_default_priors'.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_get_transform_default_priors",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic Execution and Testing -------------------
                exec_globals = {
                    'immutabledict': immutabledict,
                    'dist': dist,
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if the function is properly defined
                    if '_get_transform_default_priors' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, '_get_transform_default_priors' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "_get_transform_default_priors",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Call the function and check its output
                    priors = exec_locals['_get_transform_default_priors']()

                    # Ensure the expected keys are present in the returned immutabledict
                    expected_keys = ['carryover', 'adstock', 'hill_adstock']
                    for key in expected_keys:
                        self.assertIn(key, priors, f"Code snippet {i}: missed key '{key}' in output.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "_get_transform_default_priors",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_get_transform_default_priors",
                        "code": code,
                        "result": "failed"
                    })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write results into test_result.jsonl =============
        # Read existing test result.jsonl or ignore if not present
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for function "_get_transform_default_priors"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_get_transform_default_priors"
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