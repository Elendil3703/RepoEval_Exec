import unittest
import json
import os
import sys
from typing import Any, List
import jax.numpy as jnp

TEST_RESULT_JSONL = "test_result.jsonl"

class TestForwardZero(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        # Get the 253rd code snippet (index 252)
        cls.function_code = data[252]
        if not cls.function_code:
            raise ValueError("Expected non-empty code snippet at index 252")

    def test_forward_zero(self):
        passed_count = 0
        failed_count = 0
        results = []

        # Prepare test inputs
        emb = jnp.ones((4, 5))  # Example embedding input
        mask = jnp.array([True, False, True, False])  # Example mask

        # Prepare the execution environment
        exec_globals = {
            'jnp': jnp,
            'model': sys.modules.get('model', MockModel()),  # Mock or load actual model
            'Any': Any,
        }
        exec_locals = {}

        try:
            # Execute the function code
            exec(self.function_code, exec_globals, exec_locals)

            # Check if forward_zero function is defined
            if 'forward_zero' not in exec_locals:
                print("Function 'forward_zero' not found after execution.")
                failed_count += 1
                results.append({
                    "function_name": "forward_zero",
                    "code": self.function_code,
                    "result": "failed"
                })
                return

            # Call the function
            forward_zero = exec_locals['forward_zero']
            output = forward_zero(emb, mask)

            # Validate output
            self.assertTrue(hasattr(output, 'output'), "Output does not have attribute 'output'.")
            self.assertEqual(output.output.shape, emb.shape, "Output shape mismatch.")

            print("Test on forward_zero PASSED.")
            passed_count += 1
            results.append({
                "function_name": "forward_zero",
                "code": self.function_code,
                "result": "passed"
            })

        except Exception as e:
            print(f"Test on forward_zero FAILED with error: {e}")
            failed_count += 1
            results.append({
                "function_name": "forward_zero",
                "code": self.function_code,
                "result": "failed"
            })

        # Final test summary
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

        # Remove old records with function_name == "forward_zero"
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

class MockModel:
    class TransformerConfig:
        def __init__(self, num_heads, num_layers, key_size, mlp_hidden_size,
                     dropout_rate, causal, layer_norm, activation_function):
            self.num_heads = num_heads
            self.num_layers = num_layers
            self.key_size = key_size
            self.mlp_hidden_size = mlp_hidden_size
            self.dropout_rate = dropout_rate
            self.causal = causal
            self.layer_norm = layer_norm
            self.activation_function = activation_function

    class Transformer:
        def __init__(self, config):
            self.config = config

        def __call__(self, emb, mask):
            # Simply return an object with output attribute
            return type('Output', (object,), {'output': emb})

if __name__ == "__main__":
    unittest.main()