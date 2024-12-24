import unittest
import json
import os
import jax
import jax.numpy as jnp
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestEmbedFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[242]  # Get the 243rd JSON element (index 242)
        if not cls.code_snippet.strip():
            raise ValueError("Expected a non-empty code snippet at index 242")

    def test_embed_function(self):
        """Dynamically test the embed function in the JSON."""
        results = []
        passed_count = 0
        failed_count = 0

        code = self.code_snippet

        exec_globals = {
            'jax': jax,
            'jnp': jnp,
        }
        exec_locals = {}

        try:
            # Dynamically execute code snippet
            exec(code, exec_globals, exec_locals)

            # Check if the function exists in exec_locals
            if 'embed' not in exec_locals:
                print(f"FAILED: 'embed' not found in exec_locals.\n")
                failed_count += 1
                results.append({
                    "function_name": "embed",
                    "code": code,
                    "result": "failed"
                })
                return

            # Define dummy embedding functions
            class MockEmbedder:
                def token_embed(self, tokens):
                    return jnp.ones(tokens.shape + (3,))  # Adding a mock feature dimension

                def position_embed(self, positions):
                    return jnp.zeros(positions.shape + (3,))  # Returning zeros for position embedding

            embedder = MockEmbedder()
            tokens = jnp.array([[1, 2, 3], [4, 5, 6]])
            expected_output = jnp.ones((2, 3, 3))  # Expected sum of token and position embeddings

            # Test the embed function
            output = exec_locals['embed'](embedder, tokens)
            self.assertTrue(jnp.array_equal(output, expected_output),
                            "The embed function did not generate the expected output.")

            print("PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "embed",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "embed",
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

        # Remove old records for "embed"
        existing_records = [rec for rec in existing_records if rec.get("function_name") != "embed"]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()