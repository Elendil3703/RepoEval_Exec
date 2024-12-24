import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestTokenPosEmbed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file and extract the specified code snippet
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[184]  # Get the specified JSON element
        if not cls.code_snippet:
            raise ValueError("Expected at least one code snippet in the specified JSON array")

    def test_token_pos_embed(self):
        """Dynamically test the token_pos_embed function with various scenarios."""
        results = []  # Collect results to write to JSONL

        code = self.code_snippet

        # Static checks
        if "def token_pos_embed" not in code:
            print("FAILED, function 'token_pos_embed' not found in code.\n")
            results.append({
                "function_name": "token_pos_embed",
                "code": code,
                "result": "failed"
            })
            return

        # Dynamic execution and logic testing
        exec_globals = {
            'jnp': __import__('jax.numpy'),
            'assemble': __import__('assemble_placeholder'),  # A placeholder for the actual module
            'residual_space': Any,
            'input_space': Any,
            'indices_space': Any,
            'output_space': Any
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if the function exists
            if 'token_pos_embed' not in exec_locals:
                print("FAILED, 'token_pos_embed' not defined after execute.\n")
                results.append({
                    "function_name": "token_pos_embed",
                    "code": code,
                    "result": "failed"
                })
                return

            # Define dummy variables and modules to mock the function behavior
            dummy_tokens = exec_globals['jnp'].array([[1, 2, 3], [4, 5, 6]])
            mock_embed_modules = type('Mock', (object,), {
                'pos_embed': lambda self, indices: indices * 2
            })
            exec_globals['assemble']._make_embedding_modules = lambda **kwargs: mock_embed_modules()
            
            # Call the function and check its behavior
            pos_embed_result = exec_locals['token_pos_embed'](dummy_tokens)
            expected_result = exec_globals['jnp'].indices(dummy_tokens.shape)[-1] * 2

            # Assertions to verify the function behavior
            self.assertTrue(
                (pos_embed_result == expected_result).all(),
                "The token_pos_embed function did not return the expected result."
            )

            print("PASSED all assertions.\n")
            results.append({
                "function_name": "token_pos_embed",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"FAILED with error: {e}\n")
            results.append({
                "function_name": "token_pos_embed",
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

        # Remove old records for token_pos_embed
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "token_pos_embed"
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