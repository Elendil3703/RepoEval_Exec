import unittest
import json
import os
import sys
from typing import Callable
import rasp

TEST_RESULT_JSONL = "test_result.jsonl"

class TestMakeSortFreq(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[177]  # Get the 178th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_make_sort_freq(self):
        """Dynamically test the make_sort_freq function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Static check to ensure make_sort_freq is in the code
                if "def make_sort_freq" not in code:
                    print(f"Code snippet {i}: FAILED, 'make_sort_freq' not found in code.\n")
                    failed_count += 1
                    # Record failure
                    results.append({
                        "function_name": "make_sort_freq",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'rasp': rasp,
                }
                exec_locals = {}

                try:
                    # Dynamically execute code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if make_sort_freq is defined
                    assert 'make_sort_freq' in exec_locals, f"'make_sort_freq' not found in locals for code snippet {i}"

                    # Prepare the function
                    make_sort_freq = exec_locals['make_sort_freq']

                    # Define a test functionality for make_sort_freq
                    sort = make_sort_freq(max_seq_len=5)
                    result = sort([2, 4, 2, 1])
                    expected_result = [2, 2, 4, 1]

                    self.assertEqual(
                        result,
                        expected_result,
                        f"Code snippet {i} did not sort by frequency correctly. Expected {expected_result}, got {result}."
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "make_sort_freq",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "make_sort_freq",
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

        # Remove old records for the function
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "make_sort_freq"
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