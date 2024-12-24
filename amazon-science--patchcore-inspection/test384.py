import unittest
import json
import os
import torch
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class MockBackbone:
    def __init__(self, reach_last_layer=False):
        self.reach_last_layer = reach_last_layer

    def __call__(self, images):
        if self.reach_last_layer:
            raise LastLayerToExtractReachedException()
        return "features"

class LastLayerToExtractReachedException(Exception):
    """Mock exception to simulate reaching the last layer."""
    pass

class TestForwardFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[383]  # Get the 384th JSON element (index 383)

        cls.ForwardClass = None
        exec_globals = {
            'torch': torch,
            'LastLayerToExtractReachedException': LastLayerToExtractReachedException
        }
        exec(cls.code_snippet, exec_globals)
        cls.ForwardClass = exec_globals.get('ClassWithForward')  # Assume the class name is 'ClassWithForward'

    def test_forward_without_exception(self):
        """Test forward method without reaching last layer exception."""
        model = self.ForwardClass()
        model.backbone = MockBackbone(reach_last_layer=False)

        results = model.forward(["dummy_image"])
        self.assertEqual(results, model.outputs, "Outputs should match after forward pass without exception.")

    def test_forward_with_exception(self):
        """Test forward method with last layer exception."""
        model = self.ForwardClass()
        model.backbone = MockBackbone(reach_last_layer=True)

        results = model.forward(["dummy_image"])
        self.assertEqual(results, model.outputs, "Outputs should match after forward pass with exception.")

    @classmethod
    def tearDownClass(cls):
        results = []

        # Append results for forward function
        results.append({
            "function_name": "forward",
            "code": cls.code_snippet,
            "result": "passed"
        })

        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "forward"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()