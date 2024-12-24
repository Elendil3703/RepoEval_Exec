import unittest
import json
import sys
import re
import os
import torch

TEST_RESULT_JSONL = "test_result.jsonl"
APPROX_INVERSE_HVP = "approx_inverse_hvp"

class TestApproxInverseHVPResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[286]  # Get the 287th JSON element (index 286)
        if len(cls.code_snippet) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_approx_inverse_hvp(self):
        """Test the approx_inverse_hvp function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write into JSONL

        code = self.code_snippet
        with self.subTest(code_index=286):
            print(f"Running test for function approx_inverse_hvp...")
            # ------------------- Static Checks -------------------
            if "def approx_inverse_hvp" not in code:
                print(f"Code snippet 286: FAILED, function 'approx_inverse_hvp' not found.\n")
                failed_count += 1
                results.append({
                    "function_name": APPROX_INVERSE_HVP,
                    "code": code,
                    "result": "failed"
                })
                return
            
            # ------------------- Dynamic Execution and Logic Tests -------------------
            exec_globals = {
                'torch': torch,
            }
            exec_locals = {}

            try:
                # Dynamically execute the code snippet
                exec(code, exec_globals, exec_locals)

                # Check if approx_inverse_hvp is defined
                if APPROX_INVERSE_HVP not in exec_locals:
                    print(f"Code snippet 286: FAILED, 'approx_inverse_hvp' not found in exec_locals.\n")
                    failed_count += 1
                    results.append({
                        "function_name": APPROX_INVERSE_HVP,
                        "code": code,
                        "result": "failed"
                    })
                    return

                # Try calling the function with some simple test cases
                test_function = exec_locals[APPROX_INVERSE_HVP]
                f = lambda x: x**2  # A simple test function f

                # Define example torch tensors
                params = [torch.tensor([2.0], requires_grad=True)]
                v = [torch.tensor([1.0])]

                # Call the function
                result = test_function(v, f(params[0]), params)

                # Assertions to check basic properties
                self.assertTrue(all(isinstance(r_i, torch.Tensor) for r_i in result),
                                "The result should be a list of torch Tensors.")
                self.assertEqual(len(result), len(v),
                                 "The result should have the same length as the input vector v.")
                
                print(f"Code snippet 286: PASSED all assertions.\n")
                passed_count += 1
                results.append({
                    "function_name": APPROX_INVERSE_HVP,
                    "code": code,
                    "result": "passed"
                })
            except Exception as e:
                print(f"Code snippet 286: FAILED with error: {e}\n")
                failed_count += 1
                results.append({
                    "function_name": APPROX_INVERSE_HVP,
                    "code": code,
                    "result": "failed"
                })

        # Final summary information
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed.\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # ============= Write the test results to test_result.jsonl =============
        # Read existing test_result.jsonl (if it exists)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "approx_inverse_hvp"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != APPROX_INVERSE_HVP
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