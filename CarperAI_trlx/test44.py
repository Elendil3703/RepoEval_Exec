import unittest
import json
import sys
import os
from typing import Any, Dict

TEST_RESULT_JSONL = "test_result.jsonl"

class TestSplitKwargsFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[43]  # Access the 44th JSON element

        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON data")

    def test_split_kwargs(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Static checks: Ensure _split_kwargs function is defined
                if "_split_kwargs" not in code:
                    print(f"Code snippet {i}: FAILED, '_split_kwargs' not found in code.\n")
                    failed_count += 1
                    # Write failure record
                    results.append({
                        "function_name": "_split_kwargs",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Pattern to match the function definition
                func_pattern = r"def\s+_split_kwargs\s*\("
                if not re.search(func_pattern, code):
                    print(f"Code snippet {i}: FAILED, incorrect signature for '_split_kwargs'.\n")
                    failed_count += 1
                    # Write failure record
                    results.append({
                        "function_name": "_split_kwargs",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Dynamic execution and logic testing
                exec_globals = {
                    'Any': Any,
                    'Dict': Dict,
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if _split_kwargs exists
                    if '_split_kwargs' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, '_split_kwargs' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "_split_kwargs",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Define a mock class to simulate _supported_args
                    class MockClass:
                        _supported_args = {"arg1", "arg2"}

                    # Test _split_kwargs logic
                    def test_kwargs_function():
                        func = exec_locals['_split_kwargs']
                        supported, unsupported = func(
                            MockClass, {"arg1": 1, "arg3": 3, "arg2": 2, "arg4": 4}
                        )
                        return supported, unsupported

                    # Execute and assert the expected behavior
                    supported, unsupported = test_kwargs_function()

                    self.assertEqual(
                        supported,
                        {"arg1": 1, "arg2": 2},
                        f"Code snippet {i} did not properly separate supported kwargs.",
                    )
                    self.assertEqual(
                        unsupported,
                        {"arg3": 3, "arg4": 4},
                        f"Code snippet {i} did not properly separate unsupported kwargs.",
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "_split_kwargs",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_split_kwargs",
                        "code": code,
                        "result": "failed"
                    })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write test results to test_result.jsonl =============
        # Read existing test_result.jsonl (ignore if it doesn't exist)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "_split_kwargs"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_split_kwargs"
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