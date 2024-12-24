import unittest
import json
import os
import torch
from typing import Any

# Constants
TEST_RESULT_JSONL = "test_result.jsonl"

class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.preprocessing_modules = torch.nn.ModuleList()
        for input_dim in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

class MeanMapper(torch.nn.Module):
    def __init__(self, output_dim):
        super(MeanMapper, self).__init__()
        self.output_dim = output_dim

class TestInitFunctionality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[378]  # Get the 379th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_initialization_of_preprocessing(self):
        """Test the __init__ method of the Preprocessing class."""
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                exec_globals = {
                    'torch': torch,
                    'MeanMapper': MeanMapper,
                    '__name__': '__main__'
                }
                exec_locals = {}

                try:
                    # Dynamically execute code snippet to get Preprocessing class
                    exec(code, exec_globals, exec_locals)
                    Preprocessing = exec_locals['Preprocessing']
                    
                    # Define input dimensions and an output dimension
                    input_dims = [64, 128, 256]
                    output_dim = 10

                    # Instantiate the Preprocessing class
                    preprocessing_instance = Preprocessing(input_dims, output_dim)
                    
                    self.assertEqual(preprocessing_instance.input_dims, input_dims, f"Code snippet {i}: input_dims not set correctly.")
                    self.assertEqual(preprocessing_instance.output_dim, output_dim, f"Code snippet {i}: output_dim not set correctly.")
                    self.assertEqual(len(preprocessing_instance.preprocessing_modules), len(input_dims), f"Code snippet {i}: Incorrect number of preprocessing modules.")
                    
                    for j, module in enumerate(preprocessing_instance.preprocessing_modules):
                        self.assertIsInstance(module, MeanMapper, f"Code snippet {i}: Module {j} is not an instance of MeanMapper.")
                        self.assertEqual(module.output_dim, output_dim, f"Code snippet {i}: output_dim for module {j} not set correctly.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })

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
            if rec.get("function_name") != "__init__"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()