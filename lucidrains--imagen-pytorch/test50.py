import unittest
import json
import os
import sys
from functools import wraps

TEST_RESULT_JSONL = "test_result.jsonl"

class TestOnceDecorator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[49]  # Get the 50th JSON element (index 49)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the target JSON array")

    def test_code_snippets(self):
        """Dynamically test the 'once' decorator in the JSON code snippet."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # To collect test results for writing in JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Static check to ensure the 'once' function is defined in the snippet
                if "def once" not in code:
                    print(f"Code snippet {i}: FAILED, function 'once' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "once",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'sys': sys,
                    'wraps': wraps
                }
                exec_locals = {}

                try:
                    # Execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check for the existence of the 'once' decorator
                    if 'once' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'once' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "once",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Define a sample function to test the decorator
                    @exec_locals['once']
                    def sample_func(x):
                        return x * 2

                    # Test the decorator behavior
                    result1 = sample_func(2)
                    result2 = sample_func(3)

                    # Assert checks: only the first call should return a result
                    self.assertEqual(result1, 4, f"Code snippet {i} failed: The first call to the decorated function did not return the expected result.")
                    self.assertIsNone(result2, f"Code snippet {i} failed: The second call to the decorated function should return None, but got {result2}.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "once",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "once",
                        "code": code,
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "once"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()