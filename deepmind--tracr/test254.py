import json
import unittest
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestForwardFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[253]  # Get the specified JSON element
        if len(cls.code_snippet) < 1:
            raise ValueError("Expected at least one code snippet in the specified JSON array")

    def test_forward_function(self):
        """Test the 'forward' function from the provided code snippet."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        code = self.code_snippet
        with self.subTest(code_snippet=code):
            print("Running test for the provided code snippet...")
            exec_globals = {
                'model': Any,  # Mock 'model' as a global dependency
                'jax': Any,    # Mock 'jax' as a global dependency
                'Transformer': lambda *args, **kwargs: self.MockTransformer(),
                'TransformerConfig': lambda *args, **kwargs: kwargs,
                'gelu': lambda x: x  # Mock 'gelu'
            }
            exec_locals = {}

            try:
                # Execute the code to define 'forward'
                exec(code, exec_globals, exec_locals)

                # Check if 'forward' is defined
                if 'forward' not in exec_locals:
                    print("Code snippet FAILED, 'forward' function not found after execution.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "forward",
                        "code": code,
                        "result": "failed"
                    })
                else:
                    forward = exec_locals['forward']
                    
                    # Perform a test call with dummy inputs
                    dummy_emb = 'dummy_embedding'
                    dummy_mask = 'dummy_mask'
                    result = forward(dummy_emb, dummy_mask)

                    self.assertIsInstance(result, str, "Output of forward is not of expected type.")
                    self.assertEqual(result, 'transformer_output', "Forward function did not return expected result.")
                    
                    print("Code snippet PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "forward",
                        "code": code,
                        "result": "passed"
                    })
            except Exception as e:
                print(f"Code snippet FAILED with error: {e}\n")
                failed_count += 1
                results.append({
                    "function_name": "forward",
                    "code": code,
                    "result": "failed"
                })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for the 'forward' function
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "forward"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

    class MockTransformer:
        """A simple mock class for Transformer."""
        def __call__(self, emb, mask):
            return self

        @property
        def output(self):
            return 'transformer_output'

if __name__ == "__main__":
    unittest.main()