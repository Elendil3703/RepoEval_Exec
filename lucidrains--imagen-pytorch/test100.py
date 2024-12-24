import unittest
import json
import os
from typing import Callable, List, Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestFindFirst(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[99]  # Get the 100th JSON element (index 99)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_code_snippets(self):
        """Dynamically test the function 'find_first'."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                # Check if the 'find_first' function is defined
                if "def find_first" not in code:
                    print(f"Code snippet {i}: FAILED, function 'find_first' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "find_first",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Prepare the execution environment
                exec_globals = {
                    'Callable': Callable,
                    'List': List,
                    'Any': Any,
                }
                exec_locals = {}

                try:
                    # Execute the snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if 'find_first' was executed
                    if 'find_first' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'find_first' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "find_first",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Retrieve the tested function
                    find_first = exec_locals['find_first']

                    # Define test cases
                    test_cases = [
                        (lambda x: x > 0, [0, -1, 2, 3], 2),
                        (lambda x: x % 2 == 0, [1, 3, 5, 8, 10], 8),
                        (lambda x: x == 'hello', ['bye', 'hi', 'hello'], 'hello'),
                        (lambda x: len(x) > 3, ['a', 'abc', 'abcd'], 'abcd'),
                        (lambda x: x < 0, [3, 5, 8], None),
                    ]

                    # Verify the correctness of find_first with the test cases
                    for cond, arr, expected in test_cases:
                        result = find_first(cond, arr)
                        self.assertEqual(
                            result, expected,
                            f"Code snippet {i} failed for inputs (cond: {cond}, arr: {arr}). Expected {expected}, got {result}."
                        )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "find_first",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "find_first",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
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

        # Remove old records with the same function name
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "find_first"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test results file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()