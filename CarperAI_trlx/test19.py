import unittest
import json
import sys
import os
import torch.nn as nn
from typing import Tuple

TEST_RESULT_JSONL = "test_result.jsonl"

def findattr(model, attrs):
    """Mock findattr function to simulate finding decoder layers in a model."""
    for attr in attrs:
        if hasattr(model, attr):
            return getattr(model, attr)
    raise AttributeError("No matching decoder layers found.")

def hf_get_decoder_blocks(model: nn.Module) -> Tuple[nn.Module]:
    """Returns the decoder hidden layers of the specified model."""
    hidden_layers_attrs = (
        "h",
        "layers",
        "decoder.layers",
        "transformer.h",
        "model.decoder.layers",
        "gpt_neox.layers",
        "decoder.block",
    )
    return findattr(model, hidden_layers_attrs)

class TestHfGetDecoderBlocks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[18]  # Get the 19th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_hf_get_decoder_blocks(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                exec_globals = {
                    'nn': nn,
                    'Tuple': Tuple,
                    'findattr': findattr,
                }
                exec_locals = {}

                try:
                    exec(code, exec_globals, exec_locals)

                    if 'hf_get_decoder_blocks' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'hf_get_decoder_blocks' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "hf_get_decoder_blocks",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Create mock models with expected attributes
                    class MockModel1(nn.Module):
                        def __init__(self): self.h = nn.ModuleList()

                    class MockModel2(nn.Module):
                        def __init__(self): self.layers = nn.ModuleList()

                    # Test the function with different mocked models
                    models_and_attributes = [
                        (MockModel1(), 'h'),
                        (MockModel2(), 'layers'),
                    ]

                    for mock_model, expected_attr in models_and_attributes:
                        result = exec_locals['hf_get_decoder_blocks'](mock_model)
                        self.assertIsInstance(result, nn.ModuleList, "The result should be of type nn.ModuleList.")
                        self.assertEqual(result, getattr(mock_model, expected_attr))

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "hf_get_decoder_blocks",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "hf_get_decoder_blocks",
                        "code": code,
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write the test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Delete old records with the same function_name
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "hf_get_decoder_blocks"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()