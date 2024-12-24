import unittest
import json
import os
from typing import Tuple, Any  # 确保注入的环境中有 Tuple 和 Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGetTupleIndexFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[58]  # Get the 59th JSON element (index 58)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 59th JSON array")

    def test_safe_get_tuple_index(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # ------------------- Static Checks -------------------
                if "def safe_get_tuple_index" not in code:
                    print(f"Code snippet {i}: FAILED, 'safe_get_tuple_index' function not defined.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "safe_get_tuple_index",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic Execution and Logic Testing -------------------
                exec_globals = {
                    'Any': Any,  # Inject Any
                    'Tuple': Tuple  # Inject Tuple
                }
                exec_locals = {}

                try:
                    # Dynamic execution of code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if safe_get_tuple_index is defined after execution
                    if 'safe_get_tuple_index' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'safe_get_tuple_index' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "safe_get_tuple_index",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Access the function
                    safe_get_tuple_index = exec_locals['safe_get_tuple_index']

                    # Test cases to validate function behavior
                    test_cases = [
                        ((1, 2, 3), 1, None, 2),  # Normal case, should return 2
                        ((1, 2, 3), 3, "default", "default"),  # Out of bounds, should return "default"
                        ((), 0, "default", "default"),  # Empty tuple, should return "default"
                        ((1,), 0, None, 1),  # Single element tuple, should return 1
                        ((1, 2), -1, None, 2),  # Negative index, should return 2 (last element)
                        ((1, 2), -3, "none", "none"),  # Out of bounds negative index, should return "none"
                    ]

                    # Run test cases
                    for tup, index, default, expected in test_cases:
                        result = safe_get_tuple_index(tup, index, default)
                        assert result == expected, f"Test case failed for input ({tup}, {index}, {default}). Expected: {expected}, got: {result}"

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "safe_get_tuple_index",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "safe_get_tuple_index",
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

        # Remove old records of function safe_get_tuple_index
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "safe_get_tuple_index"
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