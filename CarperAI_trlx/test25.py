import unittest
import json
import sys
import os
import re
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCarperAITrlxResultSplitToPatterns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file and read the 25th element (index 24)
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[24]
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_split_to_patterns(self):
        """Dynamically test all code snippets in the JSON with additional checks for split_to_patterns."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Static check for function definition
                if "def split_to_patterns" not in code:
                    print(f"Code snippet {i}: FAILED, function 'split_to_patterns' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "split_to_patterns",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Regular expression to validate function signature
                func_pattern = r"def\s+split_to_patterns\s*\("
                if not re.search(func_pattern, code):
                    print(f"Code snippet {i}: FAILED, incorrect signature for 'split_to_patterns'.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "split_to_patterns",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Dynamic execution of the code snippet
                exec_globals = {'sys': sys, 'Any': Any}
                exec_locals = {}

                try:
                    exec(code, exec_globals, exec_locals)

                    # Check if split_to_patterns is defined
                    if 'split_to_patterns' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'split_to_patterns' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "split_to_patterns",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test the function with sample input
                    split_to_patterns = exec_locals['split_to_patterns']

                    # Mock functions for testing
                    def split_to_ranges(min_, max_):
                        return [min_, max_]

                    def range_to_pattern(start, stop):
                        return f"{start}-{stop}"

                    # Inject mock functions
                    exec_locals['split_to_ranges'] = split_to_ranges
                    exec_locals['range_to_pattern'] = range_to_pattern

                    # Test cases
                    result = split_to_patterns(1, 5)
                    self.assertEqual(result, ['1-1', '1-5'], f"Code snippet {i} failed test with input (1, 5).")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "split_to_patterns",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "split_to_patterns",
                        "code": code,
                        "result": "failed"
                    })

        # Test summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    existing_records.append(json.loads(line))

        existing_records = [rec for rec in existing_records if rec.get("function_name") != "split_to_patterns"]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()