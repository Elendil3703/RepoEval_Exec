import unittest
import json
import os
from typing import Callable, Tuple, Any
import jax.numpy as jnp
import haiku as hk
import sys

TEST_RESULT_JSONL = "test_result.jsonl"

class TestOneHotEmbedUnembed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[254]  # Get the 255th JSON element (index 254)
        if not cls.code_snippet:
            raise ValueError("Expected code snippet data in JSON.")

    def test_one_hot_embed_unembed(self):
        """Test _get_one_hot_embed_unembed function."""
        code = self.code_snippet
        results = []      # Collect results to write to JSONL
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests

        def exec_and_test(code):
            exec_globals = {
                'jnp': jnp,
                'hk': hk,
                'sys': sys
            }
            exec_locals = {}

            try:
                # Dynamically execute the code snippet
                exec(code, exec_globals, exec_locals)

                # Check if _get_one_hot_embed_unembed is indeed defined in the locals
                if '_get_one_hot_embed_unembed' not in exec_locals:
                    raise AssertionError("'_get_one_hot_embed_unembed' not found after execution.")

                # Retrieve the function
                _get_one_hot_embed_unembed: Callable[[int, int], Tuple[Any, Any, Any]] = exec_locals['_get_one_hot_embed_unembed']

                # Define test parameters
                vocab_size = 5
                max_seq_len = 10

                # Execute the function to get the embeddings
                token_embed, position_embed, unembed = _get_one_hot_embed_unembed(vocab_size, max_seq_len)

                # Check token_embed dimensions
                test_tokens = jnp.array([0, 1, 2])
                embedded_tokens = token_embed(test_tokens)
                self.assertEqual(embedded_tokens.shape[-1], vocab_size + max_seq_len,
                                "Token embedding dimension mismatch.")

                # Check position_embed dimensions
                test_positions = jnp.array([0, 1, 2])
                embedded_positions = position_embed(test_positions)
                self.assertEqual(embedded_positions.shape[-1], vocab_size + max_seq_len,
                                "Position embedding dimension mismatch.")

                # Test Unembed functionality
                test_embeddings = jnp.zeros((1, 3, vocab_size + max_seq_len))
                test_embeddings = test_embeddings.at[:, :, :vocab_size].set(jnp.eye(vocab_size)[test_tokens])

                unembedded_tokens = unembed(test_embeddings)
                self.assertTrue(jnp.all(unembedded_tokens == test_tokens), "Unembed function output mismatch.")

                return "passed"
            except Exception as e:
                return f"failed: {str(e)}"

        # Run the tests
        result = exec_and_test(code)
        if result == "passed":
            passed_count += 1
        else:
            failed_count += 1

        results.append({
            "function_name": "_get_one_hot_embed_unembed",
            "code": code,
            "result": result
        })

        # Print test summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed.\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # Write the results to test_result.jsonl
        # Read existing records
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for the specific function name
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_get_one_hot_embed_unembed"
        ]

        # Append new result
        existing_records.extend(results)

        # Rewrite the JSONL file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()