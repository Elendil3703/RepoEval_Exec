import unittest
import json
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class TestForwardZeroFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[237]  # Get the 238th JSON element (index 237)
        if not cls.code_snippet:
            raise ValueError("Expected a non-empty code snippet at index 237")

    def test_forward_zero_function(self):
        """Dynamically test the `forward_zero` function in the JSON snippet."""
        results = []  # To collect test results for writing to JSONL
        passed_count = 0
        failed_count = 0

        # Run tests for the code snippet
        for i, code in enumerate(self.code_snippet):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Inject necessary imports and globals
                exec_globals = {
                    'compressed_model': DummyCompressedModel,  # Replace with mock or dummy implementations
                    'model': DummyModelConfig,  # Replace with mock or dummy implementations
                    'jnp': DummyJNP(),  # Replace with mock or dummy implementations
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if forward_zero is correctly defined in the code snippet
                    if 'forward_zero' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'forward_zero' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "forward_zero",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test the forward_zero function
                    forward_zero = exec_locals['forward_zero']
                    emb = "dummy_emb"  # Replace with appropriate dummy input
                    mask = "dummy_mask"  # Replace with appropriate dummy input

                    try:
                        output = forward_zero(emb, mask)
                        assert isinstance(output, DummyOutput), f"Unexpected output type: {type(output)}"
                        print(f"Code snippet {i}: PASSED all assertions.\n")
                        passed_count += 1
                        results.append({
                            "function_name": "forward_zero",
                            "code": code,
                            "result": "passed"
                        })
                    except AssertionError as e:
                        print(f"Code snippet {i}: FAILED with assertion error: {e}\n")
                        failed_count += 1
                        results.append({
                            "function_name": "forward_zero",
                            "code": code,
                            "result": "failed"
                        })

                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "forward_zero",
                        "code": code,
                        "result": "failed"
                    })

        # Summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippet)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippet), "Test count mismatch!")

        # Write the test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for forward_zero
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "forward_zero"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

# Dummy classes to simulate the actual implementations
class DummyCompressedModel:
    class CompressedTransformer:
        def __init__(self, config):
            pass

        def __call__(self, emb, mask):
            return DummyOutput()

class DummyModelConfig:
    class TransformerConfig:
        def __init__(self, num_heads, num_layers, key_size, mlp_hidden_size, dropout_rate, causal, layer_norm, activation_function):
            pass

class DummyJNP:
    def zeros_like(self, x):
        return 0

class DummyOutput:
    pass

if __name__ == "__main__":
    unittest.main()