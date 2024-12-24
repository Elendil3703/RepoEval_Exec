import unittest
import json
import os
import torch

TEST_RESULT_JSONL = "test_result.jsonl"

class TestFeatureDimensions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[384]  # Get the 385th JSON element (index 384)
        if not cls.code_snippet:
            raise ValueError("Expected code snippet at index 384")

    def test_feature_dimensions(self):
        """Dynamically test the feature_dimensions function."""
        results = []  # Collect results to write into JSONL

        code = self.code_snippet
        
        exec_globals = {
            'torch': torch,
        }
        exec_locals = {}

        try:
            # Dynamic execution of the code snippet
            exec(code, exec_globals, exec_locals)
            
            if 'FeatureModel' not in exec_locals:
                raise ValueError("Class 'FeatureModel' not found in the executed code.")

            FeatureModel = exec_locals['FeatureModel']
            
            # Initialize a mock model with required properties
            class MockModel(FeatureModel):
                def __init__(self):
                    self.device = 'cpu'
                    self.layers_to_extract_from = ['layer1', 'layer2']
                
                def forward(self, x):
                    return {
                        'layer1': torch.ones((1, 10)),  # Simulating a tensor output
                        'layer2': torch.ones((1, 20)),  # Simulating a tensor output
                    }
            
            model = MockModel()
            input_shape = (3, 224, 224)  # Example input shape (e.g., image-like tensor)

            # Test the feature_dimensions function
            feature_dims = model.feature_dimensions(input_shape)

            # Check the returned dimensions
            self.assertEqual(feature_dims, [10, 20], "Incorrect feature dimensions returned.")

            results.append({
                "function_name": "feature_dimensions",
                "code": code,
                "result": "passed"
            })
            print(f"Code snippet: PASSED all assertions.\n")
        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}\n")
            results.append({
                "function_name": "feature_dimensions",
                "code": code,
                "result": "failed"
            })

        # Read existing records from the JSONL file if it exists
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))
        
        # Remove old records related to 'feature_dimensions'
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "feature_dimensions"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite the JSONL file with updated results
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()