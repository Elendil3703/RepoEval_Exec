import unittest
import json
import sys
import os
from typing import Any
import torch

TEST_RESULT_JSONL = "test_result.jsonl"

class TestForwardMethod(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[85]  # Get the 86th code snippet

    def test_forward_method(self):
        """Test the 'forward' method in the code snippet."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write into JSONL

        code = self.code_snippet
        print("Testing the forward method...")

        # Prepare a mock class and necessary functions
        mock_functions = """
import torch
from torch import nn

class MockClass:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.fmap_convs = [nn.Conv2d(32, 32, 3, padding=1)]

    def resize_image_to(self, fmap, target_size):
        return nn.functional.interpolate(fmap, size=target_size, mode='bilinear', align_corners=False)

    def forward(self, x, fmaps=None):
"""
        code = mock_functions + code[code.find('target_size = x.shape[-1]'):]  # Extract the function body

        exec_globals = {
            'torch': torch,
            'nn': torch.nn,
        }
        exec_locals = {}

        try:
            # Execute the code snippet with the mock class
            exec(code, exec_globals, exec_locals)

            # Instantiate the mock class and call the forward method
            mock_instance = exec_locals['MockClass']()

            # Create dummy inputs
            x = torch.randn(1, 32, 224, 224)
            fmaps = [torch.randn(1, 32, 112, 112)]

            # Call forward with fmaps enabled
            output = mock_instance.forward(x, fmaps)
            self.assertEqual(output.shape, (1, 64, 224, 224),
                             "Output shape with enabled fmaps is incorrect.")

            # Call forward with fmaps disabled
            mock_instance.enabled = False
            output = mock_instance.forward(x, fmaps)
            self.assertTrue(torch.equal(output, x),
                            "Output should match input when fmaps are disabled.")

            print("Forward method: PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "forward",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"Forward method: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "forward",
                "code": code,
                "result": "failed"
            })

        # Final statistics
        summary = f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 1\n"
        print(summary)
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # Write the test results into test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with the function_name "forward"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "forward"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)