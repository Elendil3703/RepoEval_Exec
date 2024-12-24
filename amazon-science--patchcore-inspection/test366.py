import unittest
import json
import os
import sys
import re
from unittest.mock import MagicMock, patch
from typing import Any  # Ensure Any is available in the injected environment

TEST_RESULT_JSONL = "test_result.jsonl"

class TestPredictDataloaderResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file and select the specific code snippet
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[365]  # Get the 366th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_predict_dataloader(self):
        """Dynamically test the _predict_dataloader function in the JSON."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # Static checks
                if "def _predict_dataloader" not in code:
                    print(f"Code snippet {i}: FAILED, function '_predict_dataloader' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_predict_dataloader",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Create a mock dataloader
                mock_dataloader = MagicMock()
                mock_dataloader.return_value = [{
                    "is_anomaly": MagicMock(),
                    "mask": MagicMock(),
                    "image": MagicMock()
                }]

                # Adjusting the code to mock tqdm
                code_with_mocks = code.replace("with tqdm.tqdm", "with patch('tqdm.tqdm', return_value={})")
                
                # Execution space
                exec_globals = {'patch': patch, 'MagicMock': MagicMock, 'Any': Any}
                exec_locals = {}

                try:
                    # Dynamic execution of code snippet
                    exec(code_with_mocks, exec_globals, exec_locals)

                    # Mock the class
                    class MockModel:
                        def __init__(self):
                            self.forward_modules = MagicMock()
                            self._predict = MagicMock(return_value=([0.0], [0.0]))

                    # Instantiate mock model
                    mock_model = MockModel()

                    # Execute _predict_dataloader method
                    scores, masks, labels_gt, masks_gt = exec_locals['_predict_dataloader'](mock_model, mock_dataloader)

                    # Perform assertions (you may adjust these based on expected logic)
                    self.assertEqual(scores, [0.0], "Expected score not found.")
                    self.assertEqual(masks, [0.0], "Expected mask not found.")
                    
                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "_predict_dataloader",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_predict_dataloader",
                        "code": code,
                        "result": "failed"
                    })

        # Final statistics
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

        # Remove old records
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_predict_dataloader"
        ]

        # Append new results
        existing_records.extend(results)

        # Write to test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()