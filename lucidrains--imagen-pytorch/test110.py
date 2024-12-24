import unittest
import json
import sys
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestTrainStepFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[109]  # Get the 110th JSON element (index 109)

    def test_train_step_function(self):
        """Test the train_step function in the code snippet."""
        passed_count = 0
        failed_count = 0
        results = []

        code = self.code_snippet
        with self.subTest(code=code):
            print("Running test for the code snippet containing train_step...")

            # Check if the 'train_step' function exists in the code
            if "def train_step" not in code:
                print("Code snippet: FAILED, function 'train_step' not found.\n")
                failed_count += 1
                results.append({
                    "function_name": "train_step",
                    "code": code,
                    "result": "failed"
                })
            else:
                exec_globals = {
                    'Any': Any,
                }
                exec_locals = {}

                try:
                    exec(code, exec_globals, exec_locals)

                    # Check if 'train_step' function was defined
                    if 'train_step' not in exec_locals:
                        print("Code snippet: FAILED, train_step function not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "train_step",
                            "code": code,
                            "result": "failed"
                        })
                    else:
                        print("Code snippet: PASSED all assertions.\n")
                        passed_count += 1
                        results.append({
                            "function_name": "train_step",
                            "code": code,
                            "result": "passed"
                        })

                except Exception as e:
                    print(f"Code snippet: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "train_step",
                        "code": code,
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # ============= Write the test result to test_result.jsonl =============
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
            if rec.get("function_name") != "train_step"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()