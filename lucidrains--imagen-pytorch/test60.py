import unittest
import json
import sys
import os
import torch
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestProbMaskLikeFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[59]  # Get the 60th JSON element (index 59)
        if not cls.code_snippet:
            raise ValueError("Expected code snippet at index 59 in JSON data")

    def test_prob_mask_like(self):
        """Test the 'prob_mask_like' function extracted from the code snippet."""

        # Dynamically execute the code snippet to define `prob_mask_like`
        exec_globals = {"torch": torch, "Any": Any}
        exec_locals = {}
        exec(self.code_snippet, exec_globals, exec_locals)

        # Check if prob_mask_like function is defined
        self.assertIn('prob_mask_like', exec_locals, "Function 'prob_mask_like' not found in the code snippet.")

        prob_mask_like = exec_locals['prob_mask_like']

        # Define test cases
        test_cases = [
            # Test case for prob = 1
            {
                "input": ((2, 2), 1, 'cpu'),
                "expected": torch.ones((2, 2), dtype=torch.bool)
            },
            # Test case for prob = 0
            {
                "input": ((3, 3), 0, 'cpu'),
                "expected": torch.zeros((3, 3), dtype=torch.bool)
            },
            # Test case for prob = 0.5
            {
                "input": ((4, 4), 0.5, 'cpu'),
                # Expect a tensor of shape (4, 4) with no specific content guarantee, just the dtype and device
                "expected_shape": (4, 4),
                "expected_dtype": torch.bool,
                "expected_device": 'cpu',
            }
        ]

        results = []  # Collect test results for writing to JSONL
        passed_count = 0
        failed_count = 0

        for i, case in enumerate(test_cases):
            with self.subTest(test_index=i):
                shape, prob, device = case["input"]
                expected = case.get("expected")
                
                # Execute the test case
                mask = prob_mask_like(shape, prob, device)
                
                try:
                    if expected is not None:
                        # Directly compare tensors for prob = 0 or 1
                        torch.testing.assert_close(mask, expected)
                    else:
                        # Check shape, dtype, and device for prob = 0.5
                        self.assertEqual(mask.shape, case["expected_shape"])
                        self.assertEqual(mask.dtype, case["expected_dtype"])
                        self.assertEqual(str(mask.device), case["expected_device"])
                    
                    passed_count += 1
                    results.append({
                        "function_name": "prob_mask_like",
                        "input": case["input"],
                        "result": "passed",
                    })
                except AssertionError as e:
                    print(f"Test case {i} failed: {e}")
                    failed_count += 1
                    results.append({
                        "function_name": "prob_mask_like",
                        "input": case["input"],
                        "result": "failed",
                    })

        # Assert all test cases covered
        self.assertEqual(passed_count + failed_count, len(test_cases), "Test count mismatch!")

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with same function name
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "prob_mask_like"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()