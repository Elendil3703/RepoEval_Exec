import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestRightPadDimsTo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[56]  # Get the 57th JSON element (index 56)

        if not cls.code_snippet:
            raise ValueError("Expected a code snippet in the 57th JSON array")

    def test_right_pad_dims_to(self):
        """Test the right_pad_dims_to function with various inputs."""
        passed_count = 0
        failed_count = 0
        results = []

        # Assume the existence of a test case to validate the function
        code = self.code_snippet
        exec_globals = {
            'Any': Any,
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet to define the function
            exec(code, exec_globals, exec_locals)

            right_pad_dims_to = exec_locals.get('right_pad_dims_to')

            if not right_pad_dims_to:
                raise RuntimeError("Function 'right_pad_dims_to' is not defined after executing the code snippet.")

            import torch

            # Define test cases
            test_cases = [
                (torch.rand(4, 5, 6), torch.rand(4, 5)),
                (torch.rand(3, 4), torch.rand(3, 4, 5)),
                (torch.rand(2, 3, 4), torch.rand(2, 3, 4)),
                (torch.rand(6, 1, 3), torch.rand(6)),
            ]

            # Expected outcomes
            expected_results = [
                torch.rand(4, 5, 1),
                torch.rand(3, 4, 5),
                torch.rand(2, 3, 4),
                torch.rand(6, 1, 3),
            ]

            # Test each case
            for i, (x, t) in enumerate(test_cases):
                with self.subTest(case_index=i):
                    padded_tensor = right_pad_dims_to(x, t)
                    expected_tensor = expected_results[i]

                    # Even though the tensor values differ due to randomness, we check the shape
                    self.assertEqual(
                        padded_tensor.shape, expected_tensor.shape,
                        f"Failed for test case {i}: expected shape {expected_tensor.shape}, got {padded_tensor.shape}"
                    )

                    print(f"Test case {i}: PASSED")
                    passed_count += 1
                    results.append({
                        "function_name": "right_pad_dims_to",
                        "code": code,
                        "result": "passed"
                    })

        except Exception as e:
            print(f"Test case failed with error: {e}")
            failed_count += 1
            results.append({
                "function_name": "right_pad_dims_to",
                "code": code,
                "result": "failed"
            })
        
        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(test_cases)}\n")
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

        # Remove old records
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "right_pad_dims_to"
        ]

        # Append new results
        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()