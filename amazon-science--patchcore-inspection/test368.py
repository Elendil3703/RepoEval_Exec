import unittest
import json
import os
import torch
import sys
from typing import List, Tuple, Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestPatchifyFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[367]  # Get the 368th JSON element (index 367)
        if len(cls.code_snippet) < 1:
            raise ValueError("Expected code snippet in the JSON data")

    def test_patchify(self):
        """Test the patchify function logic."""
        code = self.code_snippet
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        exec_globals = {
            'torch': torch,
            'sys': sys,
            'Any': Any,
        }
        exec_locals = {}

        try:
            # Execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if the class and function exist
            if 'TestClass' not in exec_locals:
                print("Class 'TestClass' not found in the provided code.")
                failed_count += 1
                results.append({
                    "function_name": "patchify",
                    "code": code,
                    "result": "failed"
                })
                self.fail("Class 'TestClass' not found.")
            
            # Create an instance of the class and check if the method exists
            test_instance = exec_locals['TestClass']()
            if not hasattr(test_instance, 'patchify'):
                print("Function 'patchify' not found in 'TestClass'.")
                failed_count += 1
                results.append({
                    "function_name": "patchify",
                    "code": code,
                    "result": "failed"
                })
                self.fail("Function 'patchify' not found.")

            # Test input data
            batch_size, channels, width, height = 2, 3, 8, 8
            features = torch.rand((batch_size, channels, width, height))
            
            # Call the patchify function
            output, spatial_info = test_instance.patchify(features, return_spatial_info=True)

            # Perform various checks
            expected_patch_shape = (
                batch_size * (width // test_instance.stride) * (height // test_instance.stride),
                channels,
                test_instance.patchsize,
                test_instance.patchsize,
            )
            self.assertEqual(output.shape, expected_patch_shape,
                             "Output shape did not match expected patched tensor shape.")

            self.assertEqual(len(spatial_info), 2,
                             "Spatial information should contain two elements.")
            self.assertIsInstance(spatial_info, list,
                                  "Spatial information should be a list.")

            print("Patchify function passed all assertions.")
            passed_count += 1
            results.append({
                "function_name": "patchify",
                "code": code,
                "result": "passed"
            })

        except Exception as e:
            print(f"Testing patchify function FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "patchify",
                "code": code,
                "result": "failed"
            })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 1\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # Write the test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Delete records with function_name == "patchify"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "patchify"
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