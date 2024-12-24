import unittest
import json
import sys
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGroundTruthForwardFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[238]  # Get the 239th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 239th JSON array")

    def test_forward_function(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # ------------------- Static checks -------------------
                # Check if the required components are in the snippet
                if "CompressedTransformer" not in code:
                    print(f"Code snippet {i}: FAILED, 'CompressedTransformer' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "forward",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                func_pattern = r"def\s+forward\s*\("
                if not re.search(func_pattern, code):
                    print(f"Code snippet {i}: FAILED, incorrect signature for 'forward'.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "forward",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic execution tests -------------------
                exec_globals = {
                    'sys': sys,
                    'compressed_model': type('compressed_model', (), {})(),
                    'model': type('model', (), {})(),
                    'jax': type('jax', (), {'nn': type('nn', (), {'gelu': None})()})()
                }
                exec_locals = {}

                try:
                    # Mock classes and methods
                    class MockTransformer:
                        def __init__(self, config):
                            self.config = config

                        def __call__(self, emb, mask):
                            return type('MockOutput', (), {'output': 'processed'})()

                    class MockTransformerConfig:
                        def __init__(self, num_heads, num_layers, key_size, mlp_hidden_size, dropout_rate, causal, layer_norm, activation_function):
                            pass

                    exec_globals['compressed_model'].CompressedTransformer = MockTransformer
                    exec_globals['model'].TransformerConfig = MockTransformerConfig

                    # Execute code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check forward function existence
                    if 'forward' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'forward' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "forward",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test forward function logic
                    result = exec_locals['forward'](None, None)
                    self.assertEqual(result, 'processed', f"Code snippet {i} did not process inputs correctly.")

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

        # ============= Write results to test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for the function "forward"
        existing_records = [rec for rec in existing_records if rec.get("function_name") != "forward"]

        # Append new results
        existing_records.extend(results)

        # Rewrite the test_result.jsonl file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()