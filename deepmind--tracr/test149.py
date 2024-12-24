import unittest
import json
import os
import statistics
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"


class TestRepoEval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[148]  # Get the 149th JSON element
        if not cls.code_snippet:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_mean_function(self):
        """Test the mean function for different types of inputs."""
        results = []  # Collect test results

        code = self.code_snippet
        exec_globals = {"statistics": statistics}
        exec_locals = {}

        # Dynamically execute the code snippet
        try:
            exec(code, exec_globals, exec_locals)
            if 'mean' not in exec_locals:
                raise ValueError("Function 'mean' not defined in the code snippet.")

            mean = exec_locals['mean']

            test_cases = [
                ([], 0, 0),
                ([1, 2, 3], 0, 2),
                ([True, False, True], 0, 2 / 3),
                ([10], 0, 10),
                ([100, 200, 300, 400], 0, 250),
            ]

            passed_count = 0
            failed_count = 0

            for i, (inputs, default, expected) in enumerate(test_cases):
                with self.subTest(test_case=i):
                    try:
                        result = mean(inputs, default)
                        self.assertEqual(
                            result, expected, f"Test case {i} failed: {result} != {expected}"
                        )
                        print(f"Test case {i}: PASSED")
                        passed_count += 1
                        results.append({
                            "function_name": "mean",
                            "test_case": i,
                            "inputs": inputs,
                            "default": default,
                            "expected": expected,
                            "result": "passed"
                        })
                    except Exception as e:
                        print(f"Test case {i}: FAILED with error: {e}")
                        failed_count += 1
                        results.append({
                            "function_name": "mean",
                            "test_case": i,
                            "inputs": inputs,
                            "default": default,
                            "expected": expected,
                            "error": str(e),
                            "result": "failed"
                        })

            print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(test_cases)}")
            self.assertEqual(passed_count + failed_count, len(test_cases), "Test count mismatch!")

        except Exception as e:
            print(f"Code execution failed with error: {e}")
            results.append({
                "function_name": "mean",
                "code": code,
                "error": str(e),
                "result": "failed"
            })

        # Write the test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Delete existing records for 'mean'
        existing_records = [rec for rec in existing_records if rec.get("function_name") != "mean"]

        # Append new results
        existing_records.extend(results)

        # Rewrite the JSONL file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")


if __name__ == "__main__":
    unittest.main()