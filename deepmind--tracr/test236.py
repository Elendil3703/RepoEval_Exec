import unittest
import json
import sys
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCarperAITrlxResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[235]  # Get the 236th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 236th JSON array")

    def test_code_snippets(self):
        """Dynamically test code snippets for the forward function."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                # Check if 'forward' function is present in code
                if "def forward" not in code:
                    print(f"Code snippet {i}: FAILED, function 'forward' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "forward",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Prepare mock environment for dynamic execution
                exec_globals = {
                    'compressed_model': MockCompressedModel(),  # Mocked object
                    'model': MockModel(),                      # Mocked object
                    'layer_norm': None,                        # Just a placeholder
                }
                exec_locals = {}

                try:
                    # Execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check for 'forward' function in exec_locals
                    if 'forward' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'forward' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "forward",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Call the forward function to test it
                    forward_func = exec_locals['forward']
                    
                    # Mock inputs
                    emb = "dummy_embedding"
                    mask = "dummy_mask"
                    
                    # Execute forward function
                    output = forward_func(emb, mask)
                    
                    # Verify output structure
                    self.assertEqual(output, "mock_output", 
                                     f"Code snippet {i} forward function did not return expected output.")
                    
                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "forward",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "forward",
                        "code": code,
                        "result": "failed"
                    })

        # Summary of test results
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

        # Filter out old forward records
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "forward"
        ]

        # Append new test results
        existing_records.extend(results)

        # Overwrite the JSONL file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

class MockCompressedModel:
    class CompressedTransformer:
        def __init__(self, config):
            pass
        def __call__(self, emb, mask):
            return MockOutput()

class MockModel:
    class TransformerConfig:
        def __init__(self, num_heads, num_layers, key_size, mlp_hidden_size, dropout_rate, layer_norm):
            pass

class MockOutput:
    @property
    def output(self):
        return "mock_output"

if __name__ == "__main__":
    unittest.main()