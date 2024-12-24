import unittest
import json
import sys
import os
import torch
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestQSampleFunction(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[61]  # Get the 62nd JSON element
        if not cls.code_snippet:
            raise ValueError("Expected code snippets in the JSON array")
    
    def test_q_sample_function(self):
        """Dynamically test the q_sample function implementation."""
        passed_count = 0
        failed_count = 0
        results = []  # Collected results to write into JSONL
        
        code = self.code_snippet
        exec_globals = {
            'torch': torch,
            'Any': Any
        }
        exec_locals = {}

        try:
            exec(code, exec_globals, exec_locals)
            
            # Check if 'q_sample' is defined
            if 'q_sample' not in exec_locals:
                raise ValueError("'q_sample' function not found in the provided code.")
            
            q_sample = exec_locals['q_sample']

            # Example Mock functions to replace undefined functions in the code
            def default(val, default_fn):
                return val if val is not None else default_fn()

            def right_pad_dims_to(tensor, target):
                return tensor
            
            def log_snr_to_alpha_sigma(log_snr):
                alpha = log_snr * 0.5  # Mock computation
                sigma = log_snr * 0.5  # Mock computation
                return alpha, sigma
            
            # Add mocked functions to the locals
            exec_locals['default'] = default
            exec_locals['right_pad_dims_to'] = right_pad_dims_to
            exec_locals['log_snr_to_alpha_sigma'] = log_snr_to_alpha_sigma

            # Prepare test data
            x_start = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
            t = 0.5
            noise = torch.tensor([[0.01, 0.02], [0.03, 0.04]], dtype=torch.float32)

            # Execute function
            alpha_x_start, log_snr, alpha, sigma = q_sample(x_start, t, noise)

            # Assertions to verify the function works as expected
            self.assertIsInstance(alpha_x_start, torch.Tensor, "Return value should be a torch.Tensor.")
            self.assertEqual(alpha_x_start.shape, x_start.shape, "The shape of the output tensor should match input shape.")

            print("Code snippet: PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "q_sample",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "q_sample",
                "code": code,
                "result": "failed"
            })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed.\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for q_sample
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "q_sample"
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