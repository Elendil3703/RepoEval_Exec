import unittest
import json
import os
import sys
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestNumToGroupsResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[97]  # Get the 98th JSON element (index 97)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 98th JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets related to num_to_groups function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Testing code snippet {i}...")
                # ------------------- Static Code Checks -------------------
                # Check if the function definition is present
                if "def num_to_groups" not in code:
                    print(f"Code snippet {i}: FAILED, function 'num_to_groups' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "num_to_groups",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic Execution and Logic Testing -------------------
                exec_globals = {
                    'Any': Any,
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if num_to_groups function exists after execution
                    if 'num_to_groups' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'num_to_groups' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "num_to_groups",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Function to test against
                    num_to_groups = exec_locals['num_to_groups']

                    # Tests for num_to_groups
                    test_cases = [
                        (10, 3, [3, 3, 3, 1]),
                        (15, 5, [5, 5, 5]),
                        (7, 2, [2, 2, 2, 1]),
                        (0, 1, []),
                        (1, 5, [1]),
                    ]

                    for num, divisor, expected in test_cases:
                        with self.subTest(num=num, divisor=divisor):
                            result = num_to_groups(num, divisor)
                            self.assertEqual(
                                result, expected,
                                f"Failed on num_to_groups({num}, {divisor}), expected {expected}, got {result}"
                            )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "num_to_groups",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "num_to_groups",
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

        # Remove old records for num_to_groups
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "num_to_groups"
        ]

        # Append new results
        existing_records.extend(results)

        # Write back to test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()