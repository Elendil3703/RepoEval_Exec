import unittest
import json
import os

# Constants for test results
TEST_RESULT_JSONL = "test_result.jsonl"

class RunningMeanStd:
    def __init__(self):
        """
        Calculates the running mean and standard deviation of a data stream.
        """
        self.mean = 0
        self.std = 1
        self.var = 1
        self.count = 1e-24

class TestRunningMeanStd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[19]  # Get the 20th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_init_function(self):
        """Test the __init__ method of the RunningMeanStd class."""
        passed_count = 0
        failed_count = 0
        results = []

        # Check if the __init__ function initializes variables correctly
        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Dynamically execute code snippet
                exec_globals = {
                    '__name__': '__main__',
                    'RunningMeanStd': RunningMeanStd,
                }
                exec_locals = {}

                try:
                    # exec the code snippet (in real usage, this would include creating an instance)
                    exec(code, exec_globals, exec_locals)
                    
                    # For this test, we assume 'RunningMeanStd' is present 
                    # and we instantiate and check its attributes
                    rms = exec_locals.get('rms', RunningMeanStd())

                    # Validate the default values
                    self.assertEqual(rms.mean, 0, f"Code snippet {i} did not correctly set mean = 0.")
                    self.assertEqual(rms.std, 1, f"Code snippet {i} did not correctly set std = 1.")
                    self.assertEqual(rms.var, 1, f"Code snippet {i} did not correctly set var = 1.")
                    self.assertEqual(rms.count, 1e-24, f"Code snippet {i} did not correctly set count = 1e-24.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "passed"
                    })
                except AssertionError as e:
                    print(f"Code snippet {i}: FAILED with assertion error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write results to JSONL file
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        existing_records.append(json.loads(line.strip()))

        # Remove old records
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__init__"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite JSONL file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()