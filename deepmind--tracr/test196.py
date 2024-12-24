import unittest
import json
import sys
import re
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class TestOperationFn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[195]  # Get the 196th JSON element (index 195)

    def test_operation_fn(self):
        """Dynamically test the 'operation_fn' snippet with specific logic."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        code = self.code_snippet

        with self.subTest():
            print(f"Running test for operation_fn snippet...")

            # Ensure the expected function is defined
            if "def operation_fn" not in code:
                print("Code snippet: FAILED, 'operation_fn' function not found in code.\n")
                failed_count += 1
                results.append({
                    "function_name": "operation_fn",
                    "code": code,
                    "result": "failed"
                })
            else:
                exec_globals = {
                    'input_space': {"left", "right", "up", "down"},
                    'output_space': {
                        "left": "left_vec",
                        "right": "right_vec",
                        "up": "up_vec",
                        "down": "down_vec",
                        "vector_from_basis_direction": lambda x: f"vector_{x}",
                        "null_vector": lambda: "null_vector"
                    },
                    'operation': lambda d: d if d in {"right", "left"} else "",
                }
                exec_locals = {}

                try:
                    # Execute the snippet
                    exec(code, exec_globals, exec_locals)

                    if 'operation_fn' not in exec_locals:
                        raise ValueError("'operation_fn' not found in exec_locals.")

                    operation_fn = exec_locals['operation_fn']

                    # Test: check operation_fn with different directions
                    test_cases = [
                        ("left", "vector_left"),
                        ("right", "vector_right"),
                        ("up", "null_vector"),
                        ("down", "null_vector"),
                    ]

                    for direction, expected in test_cases:
                        with self.subTest(direction=direction):
                            result = operation_fn(direction)
                            self.assertEqual(result, expected, f"Direction '{direction}' failed.")
                    
                    print("Code snippet: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "operation_fn",
                        "code": code,
                        "result": "passed"
                    })

                except Exception as e:
                    print(f"Code snippet: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "operation_fn",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed.\n")

        # ============= Write test results to test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for 'operation_fn'
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "operation_fn"
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