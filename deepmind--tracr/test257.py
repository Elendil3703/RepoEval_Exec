import unittest
import json
import sys
import re
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestEmbedUnembedFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[256]  # Get the 257th JSON element (index 256)
        if not cls.code_snippet:
            raise ValueError("Expected a valid code snippet at index 256")

    def test_embed_unembed(self):
        """Test the `embed_unembed` function logic and execution."""
        results = []  # To collect test results for JSONL

        code = self.code_snippet
        passed_count = 0
        failed_count = 0

        try:
            # Perform static checks
            if "def embed_unembed" not in code:
                raise ValueError("Function definition 'embed_unembed' not found in code.")

            # Dynamic execution check setup
            exec_globals = {
                'model': type('model', (object,), {}),
                'jax': type('jax', (object,), {'nn': type('nn', (object,), {'gelu': lambda x: x})}),
                'pad_token': 0,
                'vocab_size': 1000,
                'max_seq_len': 512,
                'self': type('MockSelf', (object,), {
                    '_get_one_hot_embed_unembed': lambda self, vs, msl: (None, None, None)
                })()
            }
            exec_locals = {}

            # Dynamic execution of the code snippet
            exec(code, exec_globals, exec_locals)
            if 'embed_unembed' not in exec_locals:
                raise ValueError("Function 'embed_unembed' not found in exec_locals after execution.")

            # Testing the function behavior with mock data
            embed_unembed = exec_locals['embed_unembed']
            mock_tokens = [1, 2, 3, 4]  # Example input tokens
            embeddings, unembeddings = embed_unembed(mock_tokens)

            # Check basic structure of returned values
            self.assertIsNotNone(embeddings, "Embeddings should not be None")
            self.assertIsNotNone(unembeddings, "Unembeddings should not be None")

            # Assuming embeddings and unembeddings should be lists or similar
            self.assertIsInstance(
                embeddings, (list, tuple),
                "Expected 'embeddings' to be a list or tuple"
            )
            self.assertIsInstance(
                unembeddings, (list, tuple),
                "Expected 'unembeddings' to be a list or tuple"
            )

            passed_count += 1
            results.append({
                "function_name": "embed_unembed",
                "code": code,
                "result": "passed"
            })
            print(f"Code snippet 256: PASSED all assertions.\n")
        except Exception as e:
            print(f"Code snippet 256: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "embed_unembed",
                "code": code,
                "result": "failed"
            })

        # Summary and results writing handling
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 1\n")

        # Read existing JSONL, if any
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for this specific function name
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "embed_unembed"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite JSONL
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()