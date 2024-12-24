import unittest
import json
import os
import torch
import torch.nn as nn
from torch.nn.modules.normalization import LayerNorm

TEST_RESULT_JSONL = "test_result.jsonl"

class TestFeedForward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[80]  # Get the 81st JSON element
        if not cls.code_snippet:
            raise ValueError("Expected code snippet in the JSON array at index 80")

    def test_feedforward(self):
        """Test the FeedForward function from the code snippet."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        code = self.code_snippet
        feedforward_func_name = 'FeedForward'

        if feedforward_func_name not in code:
            print(f"Code snippet: FAILED, function '{feedforward_func_name}' not found.\n")
            failed_count += 1
            # Writing the failed record
            results.append({
                "function_name": feedforward_func_name,
                "code": code,
                "result": "failed"
            })
        else:
            exec_globals = {
                'nn': nn,
                'LayerNorm': LayerNorm
            }
            exec_locals = {}

            try:
                # Dynamically execute the code snippet
                exec(code, exec_globals, exec_locals)

                dimensions = [64, 128, 256]
                multiples = [1, 1.5, 2]
                
                for dim in dimensions:
                    for mult in multiples:
                        with self.subTest(dim=dim, mult=mult):
                            model = exec_locals[feedforward_func_name](dim, mult)
                            self.assertIsInstance(
                                model, nn.Sequential,
                                f"Expected nn.Sequential object, got {type(model)}."
                            )
                            hidden_dim = int(dim * mult)
                            self.assertEqual(model[1].out_features, hidden_dim, "Mismatch in hidden layer dimensions.")
                            self.assertEqual(model[-1].out_features, dim, "Mismatch in final layer dimensions.")
                            
                print("Code snippet: PASSED all assertions.\n")
                passed_count += 1
                results.append({
                    "function_name": feedforward_func_name,
                    "code": code,
                    "result": "passed"
                })
            except Exception as e:
                print(f"Code snippet: FAILED with error: {e}\n")
                failed_count += 1
                results.append({
                    "function_name": feedforward_func_name,
                    "code": code,
                    "result": "failed"
                })

        # Final stats
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # Write test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Delete old records for FeedForward
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != feedforward_func_name
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