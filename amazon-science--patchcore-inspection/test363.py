import unittest
import json
import sys
import os
import torch
import numpy as np
import tqdm
from unittest.mock import MagicMock
from types import MethodType

TEST_RESULT_JSONL = "test_result.jsonl"

class TestFillMemoryBankResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file and get the correct code snippet
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[362]  # Get the 363rd JSON element (index 362)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in this JSON array")

    def test_fill_memory_bank(self):
        """Dynamically test the _fill_memory_bank function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results for writing to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                if 'def _fill_memory_bank' not in code:
                    print(f"Code snippet {i}: FAILED, '_fill_memory_bank' definition not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_fill_memory_bank",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Prepare context for execution
                exec_globals = {
                    'torch': torch,
                    'np': np,
                    'tqdm': tqdm,
                }
                exec_locals = {}

                try:
                    # Execute the code to define _fill_memory_bank
                    exec(code, exec_globals, exec_locals)
                    
                    if '_fill_memory_bank' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, '_fill_memory_bank' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "_fill_memory_bank",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Mock class instance with necessary attributes for the function
                    class MockModel:
                        device = 'cpu'
                        
                        def forward_modules(self):
                            return self

                        def to(self, _):
                            return self

                        def _embed(self, input_image):
                            return input_image * 2  # Dummy embedding function

                        def eval(self):
                            pass
                    
                    mock_instance = MockModel()
                    mock_instance.featuresampler = MagicMock()
                    mock_instance.featuresampler.run = lambda x: x + 1  # Dummy feature processing

                    mock_instance.anomaly_scorer = MagicMock()
                    mock_instance.anomaly_scorer.fit = MagicMock()

                    # Bind the method to the mock instance
                    fill_memory_bank = MethodType(exec_locals['_fill_memory_bank'], mock_instance)

                    # Prepare mock input data
                    input_data = [{'image': torch.tensor([1.0, 2.0, 3.0])}]

                    # Call the method
                    fill_memory_bank(input_data)

                    # Test that the fit method was called on anomaly_scorer
                    mock_instance.anomaly_scorer.fit.assert_called_once()
                    call_args, _ = mock_instance.anomaly_scorer.fit.call_args
                    np.testing.assert_array_equal(
                        call_args[0]['detection_features'][0],
                        np.array([3.0, 5.0, 7.0, 1.0])  # Expected transformation from mock operations
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "_fill_memory_bank",
                        "code": code,
                        "result": "passed"
                    })

                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_fill_memory_bank",
                        "code": code,
                        "result": "failed"
                    })

        # Final test count validation
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write test results to test_result.jsonl
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
            if rec.get("function_name") != "_fill_memory_bank"
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