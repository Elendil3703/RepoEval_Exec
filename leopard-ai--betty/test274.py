import unittest
import json
import os
import torch
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestConvertTensorFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[273]  # Get the 274th JSON element (index 273)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 274th JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON specific to convert_tensor."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                # ------------------- Static checks -------------------
                # Check if "convert_tensor" function is defined in the code.
                if "def convert_tensor" not in code:
                    print(f"Code snippet {i}: FAILED, 'convert_tensor' function not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "convert_tensor",
                        "code": code,
                        "result": "failed"
                    })
                    continue
                
                # ------------------- Dynamic execution and tests -------------------
                exec_globals = {
                    'torch': torch,
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if convert_tensor is defined after exec
                    if 'convert_tensor' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'convert_tensor' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "convert_tensor",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    convert_tensor = exec_locals['convert_tensor']

                    # Create a test tensor
                    test_tensor = torch.tensor([1.0, 2.0, 3.0])
                    device_cpu = torch.device('cpu')
                    device_cuda = torch.device('cuda') if torch.cuda.is_available() else device_cpu

                    # Test conversion to CPU
                    result_cpu = convert_tensor(test_tensor, device=device_cpu)
                    self.assertTrue(result_cpu.device == device_cpu, f"Code snippet {i} failed to convert to CPU device.")

                    # Test conversion to GPU (if available)
                    result_cuda = convert_tensor(test_tensor, device=device_cuda)
                    self.assertTrue(result_cuda.device == device_cuda, f"Code snippet {i} failed to convert to CUDA device.")

                    # Test conversion of non-tensor item
                    non_tensor_item = [1.0, 2.0, 3.0]
                    result_non_tensor = convert_tensor(non_tensor_item)
                    self.assertEqual(non_tensor_item, result_non_tensor, f"Code snippet {i} incorrectly modified non-tensor item.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "convert_tensor",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "convert_tensor",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write the test results to test_result.jsonl
        # Read existing test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for convert_tensor
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "convert_tensor"
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