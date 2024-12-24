import unittest
import json
import os
import re
from typing import Sequence, Union
import numpy as np

TEST_RESULT_JSONL = "test_result.jsonl"

class TestRaspEvaluate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[113]  # Get the 114th JSON element (#113 because of zero-based index)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_code_snippets(self):
        """Dynamically test the evaluate function in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Static check: Ensure 'evaluate' function is present
                if "def evaluate" not in code:
                    print(f"Code snippet {i}: FAILED, 'evaluate' function not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "evaluate",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                func_pattern = r"def\s+evaluate\s*\("
                if not re.search(func_pattern, code):
                    print(f"Code snippet {i}: FAILED, incorrect signature for 'evaluate'.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "evaluate",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Dynamic execution and testing logic
                exec_globals = {
                    'Sequence': Sequence,
                    'Union': Union,
                    'np': np,
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if evaluate function was correctly defined
                    if 'evaluate' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'evaluate' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "evaluate",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Mock-up classes and objects to use in testing
                    class MockRASPExpr:
                        pass

                    class MockRASP:
                        def evaluate(self, expr, xs):
                            return xs

                    mock_expr = MockRASPExpr()
                    xs = [[False, True], [True, True]]
                    expected_output = [[False, True], [False, True]]

                    # Instantiate the main class and call evaluate
                    mock_obj = MockRASP()
                    eval_func = exec_locals['evaluate']
                    eval_func = eval_func.__get__(mock_obj, MockRASP)

                    result = eval_func(mock_expr, xs)

                    # Test the returned output
                    self.assertEqual(
                        result,
                        expected_output,
                        f"Code snippet {i} did not return the expected output."
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "evaluate",
                        "code": code,
                        "result": "passed"
                    })

                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "evaluate",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write the test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "evaluate"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "evaluate"
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