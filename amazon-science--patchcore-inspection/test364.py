import unittest
import json
import os
from typing import Any  # Ensure Any is available in the injected environment
import torch

TEST_RESULT_JSONL = "test_result.jsonl"

class TestImageToFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[363]  # Get the 364th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 364th JSON array")

    def test_image_to_features(self):
        """Dynamically test _image_to_features function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to be written to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Check for required function definition
                if "def _image_to_features" not in code:
                    print(f"Code snippet {i}: FAILED, '_image_to_features' function definition not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_image_to_features",
                        "code": code,
                        "result": "failed"
                    })
                    continue
                
                # Dynamic execution and testing
                exec_globals = {
                    'torch': torch,
                }
                exec_locals = {}

                try:
                    # Execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Create a mock class with device and _embed method
                    class MockClass:
                        device = torch.device('cpu')
                        
                        def _embed(self, input_image):
                            return torch.tensor([1.0, 2.0, 3.0])  # Mocked feature vector

                    # Instantiate the mock class
                    mock_instance = MockClass()

                    # Test _image_to_features function
                    test_image = torch.tensor([1.0, 2.0, 3.0])  # Mocked input image
                    expected_output = mock_instance._embed(test_image.to(torch.float).to(mock_instance.device))

                    # Bind the method to the mock instance
                    _image_to_features = exec_locals['_image_to_features'].__get__(mock_instance)

                    # Run the test
                    output = _image_to_features(test_image)
                    self.assertTrue(torch.equal(output, expected_output), 
                                    f"Code snippet {i}: Output did not match expected features.\n")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "_image_to_features",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_image_to_features",
                        "code": code,
                        "result": "failed"
                    })

        # Test summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Update test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for _image_to_features
        existing_records = [rec for rec in existing_records if rec.get("function_name") != "_image_to_features"]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()