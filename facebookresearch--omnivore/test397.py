import unittest
import json
import sys
import re
import os
from typing import Any, List, Callable
import torch
import logging

TEST_RESULT_JSONL = "test_result.jsonl"

class TestLoadCheckpointAndApplyKernels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[396]  # Get the 397th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the first JSON array")

    def test_load_checkpoint_and_apply_kernels(self):
        """Dynamically test load_checkpoint_and_apply_kernels with sample inputs and validation."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # ------------------- Static Checks -------------------
                if "def load_checkpoint_and_apply_kernels" not in code:
                    print(f"Code snippet {i}: FAILED, function 'load_checkpoint_and_apply_kernels' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "load_checkpoint_and_apply_kernels",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                func_pattern = r"def\s+load_checkpoint_and_apply_kernels\s*\("
                if not re.search(func_pattern, code):
                    print(f"Code snippet {i}: FAILED, incorrect signature for 'load_checkpoint_and_apply_kernels'.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "load_checkpoint_and_apply_kernels",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic Execution and Testing -------------------
                exec_globals = {
                    'torch': torch,
                    'logging': logging,
                    'Any': Any,
                    'List': List,
                    'Callable': Callable,
                }
                exec_locals = {}

                try:
                    # Mock the g_pathmgr
                    class MockPathMgr:
                        @staticmethod
                        def exists(path):
                            return True

                        @staticmethod
                        def open(path, mode):
                            from io import BytesIO
                            return BytesIO()  # Return a dummy stream for the checkpoint

                    exec_globals['g_pathmgr'] = MockPathMgr()

                    # Dummy state dict for testing
                    dummy_state_dict = {'layer1.weight': torch.tensor([1.0])}
                    def mock_load(f, map_location=None):
                        return {'state_dict': dummy_state_dict}

                    exec_globals['torch'].load = mock_load

                    exec(code, exec_globals, exec_locals)

                    # Check if load_checkpoint_and_apply_kernels exists
                    if 'load_checkpoint_and_apply_kernels' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'load_checkpoint_and_apply_kernels' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "load_checkpoint_and_apply_kernels",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Define a sample kernel
                    def sample_kernel(state_dict):
                        # Modify state_dict
                        state_dict['layer1.bias'] = torch.tensor([0.0])
                        return state_dict

                    # Call the function
                    result_state_dict = exec_locals['load_checkpoint_and_apply_kernels'](
                        "dummy/path/checkpoint.pth",
                        checkpoint_kernels=[sample_kernel]
                    )

                    # Validate the output
                    self.assertIn('layer1.weight', result_state_dict, f"Code snippet {i} did not preserve 'layer1.weight'.")
                    self.assertIn('layer1.bias', result_state_dict, f"Code snippet {i} did not apply the kernel correctly.")
                    
                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "load_checkpoint_and_apply_kernels",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "load_checkpoint_and_apply_kernels",
                        "code": code,
                        "result": "failed"
                    })

        # Summarize test results
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write the results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name "load_checkpoint_and_apply_kernels"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "load_checkpoint_and_apply_kernels"
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