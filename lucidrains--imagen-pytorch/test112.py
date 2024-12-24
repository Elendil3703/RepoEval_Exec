import unittest
import json
import os
import sys
from unittest.mock import MagicMock, patch
from torch import nn

TEST_RESULT_JSONL = "test_result.jsonl"

class MockModel:
    def __init__(self, use_ema, ema_unet_being_trained_index=None):
        self.use_ema = use_ema
        self.ema_unet_being_trained_index = ema_unet_being_trained_index
        self.device = 'cuda'
        self.unets = nn.ModuleList([nn.Linear(1, 1) for _ in range(3)])
        self.ema_unets = nn.ModuleList([nn.Linear(1, 1) for _ in range(3)])
    
    def validate_unet_number(self, unet_number):
        return unet_number if unet_number is not None else 1

class TestCarperAITrlxGetEMAUNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[111]  # Get the 112th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON data")

    def test_get_ema_unet(self):
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                if "def get_ema_unet" not in code:
                    print(f"Code snippet {i}: FAILED, 'get_ema_unet' function not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "get_ema_unet",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {}
                exec_locals = {}

                try:
                    exec(code, exec_globals, exec_locals)
                    get_ema_unet = exec_locals['get_ema_unet']

                    # Mock a model object
                    mock_model = MockModel(use_ema=True, ema_unet_being_trained_index=0)
                    mock_model.get_ema_unet = get_ema_unet.__get__(mock_model, MockModel)
                    
                    # Test the function when ema is used
                    ema_unet = mock_model.get_ema_unet(2)
                    
                    self.assertTrue(isinstance(ema_unet, nn.Module), f"Code snippet {i} did not return a nn.Module.")
                    self.assertEqual(mock_model.ema_unet_being_trained_index, 1, f"Code snippet {i} did not update 'ema_unet_being_trained_index' correctly.")

                    # Test the function without using ema
                    mock_model_no_ema = MockModel(use_ema=False)
                    mock_model_no_ema.get_ema_unet = get_ema_unet.__get__(mock_model_no_ema, MockModel)
                    
                    self.assertIsNone(mock_model_no_ema.get_ema_unet(2), f"Code snippet {i} should return None when 'use_ema' is False.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "get_ema_unet",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "get_ema_unet",
                        "code": code,
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        existing_records = [rec for rec in existing_records if rec.get("function_name") != "get_ema_unet"]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()