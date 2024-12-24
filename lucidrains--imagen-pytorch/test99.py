import unittest
import json
import os
import torch
import numpy as np
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class MockModel:
    def __init__(self, device, cast_half_at_training=False):
        self.device = device
        self.cast_half_at_training = cast_half_at_training

def fn(model, *args, **kwargs):
    # Mock function that mimics a forward pass
    return sum(arg.sum().item() for arg in args if isinstance(arg, torch.Tensor))

def exists(val):
    return val is not None

class TestInnerFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the specified JSON file and extract the 99th code snippet
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[98]  # Get the 99th JSON element

    def test_inner_function(self):
        """Test the `inner` function for expected behavior."""
        passed_count = 0
        failed_count = 0
        results = []  # Collect results to write to JSONL

        # Dynamically execute the code snippet
        exec_globals = {
            'torch': torch,
            'np': np,
            'exists': exists,
            'fn': fn,
            'cast_fp16': True,  # Simulation of the ground truth environment
            'Any': Any
        }
        exec_locals = {}
        exec(self.code_snippet, exec_globals, exec_locals)

        # Retrieve the `inner` function
        inner_func = exec_locals.get('inner')

        # Test cases
        test_cases = [
            (MockModel('cpu'), (np.array([1.0, 2.0]),), {}, 'cpu', False, torch.float32),
            (MockModel('cuda'), (torch.tensor([1.0, 2.0]),), {'_cast_device': False}, 'cpu', False, torch.float32),
            (MockModel('cuda', cast_half_at_training=True), (torch.tensor([1.0, 2.0]),), {}, 'cuda', True, torch.float16),
        ]

        for i, (model, args, kwargs, expected_device, should_cast_fp16, expected_dtype) in enumerate(test_cases):
            with self.subTest(test_case=i):
                try:
                    # Call the `inner` function
                    result = inner_func(model, *args, **kwargs)

                    # Check the types and devices of the result tensors
                    for arg in args:
                        if isinstance(arg, np.ndarray):
                            continue
                        self.assertIsInstance(arg, torch.Tensor)
                        self.assertEqual(arg.device.type, expected_device)
                        self.assertEqual(arg.dtype, expected_dtype)

                    passed_count += 1
                    results.append({
                        "function_name": "inner",
                        "test_case": i,
                        "result": "passed"
                    })
                except Exception as e:
                    failed_count += 1
                    results.append({
                        "function_name": "inner",
                        "test_case": i,
                        "result": "failed",
                        "error": str(e)
                    })

        # Overall statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(test_cases)}\n")
        self.assertEqual(passed_count + failed_count, len(test_cases), "Test count mismatch!")

        # Write the results to a JSONL file
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for the "inner" function
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "inner"
        ]

        # Prepare new results
        existing_records.extend(results)

        # Write to test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()