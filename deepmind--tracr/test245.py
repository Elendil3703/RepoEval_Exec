import unittest
import json
import os
import sys
import haiku as hk
import numpy as np

TEST_RESULT_JSONL = "test_result.jsonl"

class TestLayerNormFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[244]  # Get the 245th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 245th JSON array")

    def test_layer_norm(self):
        """Dynamically test all code snippets related to layer_norm."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                if "hk.LayerNorm" not in code:
                    print(f"Code snippet {i}: FAILED, 'hk.LayerNorm' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "layer_norm",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Dynamic execution and testing logic
                exec_globals = {
                    'sys': sys,
                    'hk': hk,
                    'np': np,
                }
                exec_locals = {}

                try:
                    # Define a dummy configuration and input for testing
                    class Config:
                        layer_norm = True

                    config = Config()
                    x = np.random.randn(3, 3).astype(np.float32)
                    
                    # Inject variables for execution
                    exec_locals['self'] = self
                    exec_locals['config'] = config
                    exec_locals['x'] = x

                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Retrieve the local variables after execution
                    layer_norm_result = exec_locals['x']

                    # Assert the shape remains the same
                    self.assertEqual(x.shape, layer_norm_result.shape, f"Code snippet {i} changed the shape of input.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "layer_norm",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "layer_norm",
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

        # Remove old records with the same function name
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "layer_norm"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()