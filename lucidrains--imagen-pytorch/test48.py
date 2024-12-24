import unittest
import json
import os
from functools import wraps
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestLucidrainsImagenPytorchResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[47]  # Get the 48th JSON element

    def test_maybe_function(self):
        """Test the correctness of the 'maybe' function."""
        passed_count = 0
        failed_count = 0
        results = []

        code = self.code_snippet

        # Inject required elements
        exec_globals = {
            'wraps': wraps,
            'Any': Any,
            'exists': lambda x: x is not None  # Mocking exists
        }
        exec_locals = {}

        try:
            # Execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if 'maybe' is defined
            if 'maybe' not in exec_locals:
                print(f"Failed: 'maybe' not defined in the executed code.")
                failed_count += 1
                results.append({
                    "function_name": "maybe",
                    "code": code,
                    "result": "failed"
                })
            else:
                # Test cases for the maybe function
                maybe_fn = exec_locals['maybe']

                def sample_function(x):
                    return x * 2

                # Wrapping the sample function with maybe
                wrapped_fn = maybe_fn(sample_function)

                # Test when input exists
                self.assertEqual(wrapped_fn(5), 10, "Failed test when input exists.")
                # Test when input is None
                self.assertIsNone(wrapped_fn(None), "Failed test when input does not exist.")

                print("All tests passed for 'maybe' function.")
                passed_count += 1
                results.append({
                    "function_name": "maybe",
                    "code": code,
                    "result": "passed"
                })

        except Exception as e:
            print(f"Exception occurred during testing: {e}")
            failed_count += 1
            results.append({
                "function_name": "maybe",
                "code": code,
                "result": "failed"
            })

        # Log test results
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed\n")

        # ============= Write test results to test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for "maybe"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "maybe"
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