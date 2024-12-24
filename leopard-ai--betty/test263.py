import unittest
import json
import sys
import re
import os
import torch
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestTrainingStepExec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[262]  # Get the 263rd JSON element
        if len(cls.code_snippet) < 1:
            raise ValueError("Expected at least one code snippet in the 263rd JSON array")

    def test_training_step_exec(self):
        """Dynamically test the 'training_step_exec' function from the code snippet."""
        passed_count = 0
        failed_count = 0
        results = []

        # Extract the code
        code = self.code_snippet

        # Test if torch and cuda are imported
        if "torch" not in code or "amp" not in code:
            print(f"Code snippet 262: FAILED, torch or torch.cuda.amp not found in code.\n")
            failed_count += 1
            results.append({
                "function_name": "training_step_exec",
                "code": code,
                "result": "failed"
            })
        else:
            exec_globals = {
                'torch': torch,
                'Any': Any,  # Inject Any
            }
            exec_locals = {}

            try:
                # Dynamically execute the code snippet
                exec(code, exec_globals, exec_locals)

                # Check if training_step_exec is defined
                if 'training_step_exec' not in exec_locals:
                    print(f"Code snippet 262: FAILED, 'training_step_exec' not found in exec_locals.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "training_step_exec",
                        "code": code,
                        "result": "failed"
                    })
                else:
                    # Mock class to test the function
                    class MockModel:
                        def _is_default_fp16(self):
                            return False

                        def training_step(self, batch):
                            return "processed_batch"

                        training_step_exec = exec_locals['training_step_exec']

                    model = MockModel()
                    result = model.training_step_exec("fake_batch")

                    self.assertEqual(result, "processed_batch", "The function did not return the expected output.")

                    print(f"Code snippet 262: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "training_step_exec",
                        "code": code,
                        "result": "passed"
                    })
            except Exception as e:
                print(f"Code snippet 262: FAILED with error: {e}\n")
                failed_count += 1
                results.append({
                    "function_name": "training_step_exec",
                    "code": code,
                    "result": "failed"
                })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed.\n")

        # Write the test results to test_result.jsonl
        # Read existing test results if file exists
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for the function "training_step_exec"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "training_step_exec"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite the test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")


if __name__ == "__main__":
    unittest.main()