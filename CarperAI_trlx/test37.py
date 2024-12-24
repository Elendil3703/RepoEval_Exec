import unittest
import json
import sys
import os
from copy import deepcopy
from torch import nn
from typing import Any, Type  # Importing required types

TEST_RESULT_JSONL = "test_result.jsonl"

class TestInitFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[36]  # Get the 37th JSON element (index 36)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 37th JSON array")

    def test_init_function(self):
        """Dynamically test __init__ function in the JSON."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Gather test results to write to JSONL

        def make_head(hidden_size, output_size, dtype):
            """Helper function to simulate `make_head` as used in __init__."""
            return nn.Linear(hidden_size, output_size, dtype=dtype)

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Test Environment preparation
                exec_globals = {
                    'nn': nn,
                    'make_head': make_head,
                    'deepcopy': deepcopy,
                }
                exec_locals = {}

                try:
                    # Execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if __init__ actually exists in the executed code
                    if '__init__' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, '__init__' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "__init__",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Now create a mock class to test the __init__ code
                    class MockModel:
                        # Inject the tested __init__ method
                        __init__ = exec_locals['__init__']

                    # Create the object with test parameters
                    model_instance = MockModel(
                        hidden_size=32,
                        vocab_size=1000,
                        two_qs=True,
                        alpha=0.5,
                        dtype=float
                    )

                    # Assertions to check each attribute and logic in __init__
                    self.assertEqual(model_instance.hidden_size, 32, "hidden_size attribute mismatch")
                    self.assertEqual(model_instance.vocab_size, 1000, "vocab_size attribute mismatch")
                    self.assertEqual(model_instance.two_qs, True, "two_qs attribute mismatch")
                    self.assertEqual(model_instance.alpha, 0.5, "alpha attribute mismatch")

                    # Check if v_head is created and has correct attributes
                    self.assertIsInstance(model_instance.v_head, nn.Linear, "v_head should be a nn.Linear instance")
                    self.assertEqual(model_instance.v_head.out_features, 1, "v_head output features mismatch")

                    # Check q_heads and target_q_heads
                    expected_qs_count = 2 if model_instance.two_qs else 1
                    self.assertEqual(len(model_instance.q_heads), expected_qs_count, "Mismatch in q_heads count")
                    self.assertEqual(len(model_instance.target_q_heads), expected_qs_count, "Mismatch in target_q_heads count")

                    for target_q_head in model_instance.target_q_heads:
                        self.assertFalse(target_q_head.requires_grad, "target_q_heads should not require grad")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })

        # Final Summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Filter out old records for the specific function
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__init__"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite the JSONL file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()