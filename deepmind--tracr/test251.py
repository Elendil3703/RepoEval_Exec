import unittest
import json
import sys
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestForwardFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[250]  # Get the 251st JSON element

        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_forward_function(self):
        """Dynamically test forward function code snippets."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results for writing to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # Static checks
                if "def forward" not in code:
                    print(f"Code snippet {i}: FAILED, function 'forward' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "forward",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Dynamic execution and logical testing
                exec_globals = {
                    'sys': sys,
                    'Any': Any,  # Inject Any
                    'model': self.mock_model(),  # Provide a mock model context
                    'layer_norm': None,  # Mock layer_norm parameter
                }
                exec_locals = {}

                try:
                    # Execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if the forward function exists
                    if 'forward' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'forward' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "forward",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Run the forward function with mocked inputs
                    emb_mock = [0] * 10  # Example mock embedding
                    mask_mock = [1] * 10  # Example mock mask

                    try:
                        output = exec_locals['forward'](emb_mock, mask_mock)
                        # Assuming a mock output structure here
                        self.assertTrue(hasattr(output, 'output'), "Output missing 'output' attribute")

                        print(f"Code snippet {i}: PASSED all assertions.\n")
                        passed_count += 1
                        results.append({
                            "function_name": "forward",
                            "code": code,
                            "result": "passed"
                        })
                    except Exception as e:
                        print(f"Code snippet {i}: FAILED during execution with error: {e}\n")
                        failed_count += 1
                        results.append({
                            "function_name": "forward",
                            "code": code,
                            "result": "failed"
                        })

                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "forward",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "forward"
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

    def mock_model(self):
        class MockTransformerConfig:
            def __init__(self, **kwargs):
                pass

        class MockTransformer:
            def __init__(self, config):
                pass

            def __call__(self, emb, mask):
                class Output:
                    output = "mock_output"
                return Output()

        return type('Model', (), {'Transformer': MockTransformer, 'TransformerConfig': MockTransformerConfig})()

if __name__ == "__main__":
    unittest.main()