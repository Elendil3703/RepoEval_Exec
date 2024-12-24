import unittest
import json
import os
from typing import Any, Callable, Dict, List  

TEST_RESULT_JSONL = "test_result.jsonl"

class DummyOptimizer:
    def __init__(self):
        self.param_groups = [{}]

class TestGroundTruthFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[412]  # Get the 413th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 413th JSON array")

    def test_step_schedulers(self):
        """Test the step_schedulers function with custom scenarios."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                if "def step_schedulers" not in code:
                    print(f"Code snippet {i}: FAILED, function 'step_schedulers' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "step_schedulers",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'Any': Any,
                    'Callable': Callable,
                    'Dict': Dict,
                    'List': List,
                    'DummyOptimizer': DummyOptimizer
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if step_schedulers is in exec_locals
                    if 'step_schedulers' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'step_schedulers' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "step_schedulers",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Create an instance of a dummy class and set the method
                    class TestClass:
                        def __init__(self):
                            self.schedulers = [
                                {"lr": lambda where: where * 0.1, "momentum": lambda where: 0.9 - where * 0.1}
                            ]
                            self.optimizer = DummyOptimizer()

                        step_schedulers = exec_locals['step_schedulers']

                    test_instance = TestClass()
                    test_instance.step_schedulers(0.5)

                    # Assertions after calling step_schedulers
                    self.assertEqual(test_instance.optimizer.param_groups[0]["lr"], 0.05)
                    self.assertEqual(test_instance.optimizer.param_groups[0]["momentum"], 0.85)

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "step_schedulers",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "step_schedulers",
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

        # Remove old records for function_name == "step_schedulers"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "step_schedulers"
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