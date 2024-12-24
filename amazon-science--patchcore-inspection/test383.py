import unittest
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

TEST_RESULT_JSONL = "test_result.jsonl"

class TestConvertToSegmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[382]  # Get the 383rd JSON element (index 382)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the chosen JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                # Check for presence of convert_to_segmentation
                if "def convert_to_segmentation" not in code:
                    print(f"Code snippet {i}: FAILED, function 'convert_to_segmentation' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "convert_to_segmentation",
                        "code": code,
                        "result": "failed"
                    })
                    continue
                    
                # Prepare execution environment
                exec_globals = {
                    'np': np,
                    'torch': torch,
                    'F': F,
                    'ndimage': ndimage,
                }
                exec_locals = {}

                try:
                    # Dynamically execute code snippet
                    exec(code, exec_globals, exec_locals)
                    
                    # Check if function is present after exec
                    if 'convert_to_segmentation' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'convert_to_segmentation' not found after exec.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "convert_to_segmentation",
                            "code": code,
                            "result": "failed"
                        })
                        continue
                    
                    # Define a mock class with expected attributes
                    class MockModel:
                        def __init__(self, device='cpu', target_size=(256, 256), smoothing=1.0):
                            self.device = device
                            self.target_size = target_size
                            self.smoothing = smoothing

                        convert_to_segmentation = exec_locals['convert_to_segmentation']
                    
                    # Initialize mock model
                    model = MockModel()
                    
                    # Prepare test data
                    patch_scores = np.random.rand(10, 128, 128).astype(np.float32)
                    
                    # Call the function
                    segmentation_results = model.convert_to_segmentation(patch_scores)
                    
                    # Validate output type and shape
                    self.assertIsInstance(segmentation_results, list, f"Code snippet {i}: Output is not a list")
                    self.assertEqual(len(segmentation_results), 10, f"Code snippet {i}: Output list length is incorrect")

                    for seg in segmentation_results:
                        self.assertIsInstance(seg, np.ndarray, f"Code snippet {i}: Segment is not a numpy array")
                        self.assertEqual(seg.shape, model.target_size, f"Code snippet {i}: Segment shape is incorrect")
                    
                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "convert_to_segmentation",
                        "code": code,
                        "result": "passed"
                    })
                    
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "convert_to_segmentation",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
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

        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "convert_to_segmentation"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()