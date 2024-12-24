import unittest
import json
import os
from typing import Any
import torch

TEST_RESULT_JSONL = "test_result.jsonl"

class TestPredictFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[364]  # Get the 365th JSON element (zero-indexed)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_predict_function(self):
        """Dynamically test all code snippets in the JSON for predict function implementation."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Check for presence of predict function in the code
                if "def predict(" not in code:
                    print(f"Code snippet {i}: FAILED, 'predict' function not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "predict",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'torch': torch,
                    'Any': Any  
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code
                    exec(code, exec_globals, exec_locals)

                    # Check predict function existence
                    if 'predict' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'predict' function not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "predict",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Create a mock object to test predict method
                    class MockModel:
                        def _predict_dataloader(self, data):
                            return "dataloader"

                        def _predict(self, data):
                            return "data"

                    # Bind the predict function to a MockModel instance
                    mock_instance = MockModel()
                    setattr(mock_instance, 'predict', exec_locals['predict'].__get__(mock_instance))

                    # Test with DataLoader
                    mock_dataloader = torch.utils.data.DataLoader([1, 2, 3])
                    dataloader_result = mock_instance.predict(mock_dataloader)
                    self.assertEqual(dataloader_result, "dataloader", f"Code snippet {i} dataloader prediction failed.")

                    # Test with non-DataLoader input
                    non_dataloader_data = [1, 2, 3]
                    data_result = mock_instance.predict(non_dataloader_data)
                    self.assertEqual(data_result, "data", f"Code snippet {i} data prediction failed.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "predict",
                        "code": code,
                        "result": "passed"
                    })

                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "predict",
                        "code": code,
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for predict function
        existing_records = [rec for rec in existing_records if rec.get("function_name") != "predict"]
        
        # Append new results
        existing_records.extend(results)

        # Rewrite the test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()