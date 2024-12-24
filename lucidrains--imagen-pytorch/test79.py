import unittest
import json
import sys
import re
import os
import torch
import torch.nn as nn
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestRepoEvalResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[78]  # Get the 79th JSON element (index 78)
        if len(cls.code_snippet) < 1:
            raise ValueError("Expected valid code snippet at index 78")

    def test_code_snippet(self):
        """Test the __init__ method of a neural network component."""
        
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to be written to JSONL

        code = self.code_snippet

        try:
            # Prepare for dynamic execution
            exec_globals = {
                'nn': nn
            }
            exec_locals = {}

            # Dynamically execute provided code
            exec(code, exec_globals, exec_locals)

            # Check if the __init__ method exists in exec_locals
            if '__init__' not in code:
                print(f"Code snippet: FAILED, '__init__' not found in code.\n")
                failed_count += 1
                results.append({
                    "function_name": "__init__",
                    "code": code,
                    "result": "failed"
                })
            else:
                # Test instantiation of the class and its properties
                dim_in = 16
                dim_out = 32
                nn_module = exec_locals['NNModule'](dim_in=dim_in, dim_out=dim_out)  # Assume class is named NNModule
                
                self.assertIsInstance(nn_module.to_k, nn.Conv2d,
                                      "to_k should be an instance of nn.Conv2d")

                self.assertEqual(nn_module.to_k.in_channels, dim_in,
                                 "to_k should have the specified input dimension")

                self.assertEqual(nn_module.to_k.out_channels, 1,
                                 "to_k should have 1 output channel")

                self.assertEqual(nn_module.net[0].in_channels, dim_in,
                                 "The first Conv2d in net should have the correct input dimension")
                
                hidden_dim = max(3, dim_out // 2)
                self.assertEqual(nn_module.net[0].out_channels, hidden_dim,
                                 "The first Conv2d in net should have the calculated hidden dimension")

                self.assertIsInstance(nn_module.net[1], nn.SiLU,
                                      "The second layer in net should be a SiLU activation")

                self.assertEqual(nn_module.net[2].in_channels, hidden_dim,
                                 "The third Conv2d in net should have the correct input dimension")

                self.assertEqual(nn_module.net[2].out_channels, dim_out,
                                 "The third Conv2d in net should have the specified output dimension")

                self.assertIsInstance(nn_module.net[3], nn.Sigmoid,
                                      "The fourth layer in net should be a Sigmoid activation")

                print(f"Code snippet: PASSED all assertions.\n")
                passed_count += 1
                results.append({
                    "function_name": "__init__",
                    "code": code,
                    "result": "passed"
                })

        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "__init__",
                "code": code,
                "result": "failed"
            })
        
        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 1\n")
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

        # Remove old records for __init__
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__init__"
        ]

        # Append new results
        existing_records.extend(results)

        # Re-write test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()