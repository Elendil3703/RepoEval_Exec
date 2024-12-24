import unittest
import json
import sys
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"
    
class TestCastTuple(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[94]  # Get the 95th JSON element

        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # ------------------- Static check -------------------
                if "def cast_tuple" not in code:
                    print(f"Code snippet {i}: FAILED, 'cast_tuple' function not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "cast_tuple",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic execution -------------------
                exec_globals = {
                    'sys': sys,
                    'Any': Any,  # Inject Any
                }
                exec_locals = {}

                try:
                    # Dynamically execute code snippet
                    exec(code, exec_globals, exec_locals)

                    # Get the defined function
                    cast_tuple = exec_locals.get('cast_tuple')
                    if not cast_tuple:
                        print(f"Code snippet {i}: FAILED, 'cast_tuple' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "cast_tuple",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # ------------------- Logic Testing -------------------
                    # Test cases
                    test_cases = [
                        (1, 1, (1,)),  # Single integer
                        ([1, 2], 1, (1, 2)),  # List to tuple
                        ("a", 3, ("a", "a", "a")),  # Singles to multiple tuple
                        ((5,), 1, (5,)),  # Already a tuple
                        (None, 2, (None, None)),  # None input
                        ([1, 2, 3], 1, (1, 2, 3)),  # List to tuple
                    ]

                    for val, length, expected in test_cases:
                        result = cast_tuple(val, length)
                        self.assertEqual(
                            result, 
                            expected, 
                            f"Code snippet {i}: FAILED, expected {expected} but got {result} for {val} and length {length}."
                        )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "cast_tuple",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "cast_tuple",
                        "code": code,
                        "result": "failed"
                    })

        # Test summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # =========== Write test results into test_result.jsonl ===========
        # Read existing test_result.jsonl (if exists)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "cast_tuple"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "cast_tuple"
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