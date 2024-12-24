import unittest
import json
import os
import torch
from torch import nn
from typing import Any  # Ensure Any is available
from einops.layers.torch import Rearrange

TEST_RESULT_JSONL = "test_result.jsonl"

class PerceiverAttention(nn.Module):
    def __init__(self, dim, dim_head, heads):
        super().__init__()
        # Dummy implementation for the sake of completeness
        pass

class FeedForward(nn.Module):
    def __init__(self, dim, mult):
        super().__init__()
        # Dummy implementation for the sake of completeness
        pass

class LayerNorm(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(dim)

class TestInitMethod(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[65]  # Get the 66th JSON element (index 65)

        if len(cls.code_snippet) < 1:
            raise ValueError("Expected a code snippet in the 66th JSON array")

        # Extract init_code for testing
        cls.init_code = cls.code_snippet['init']

    def test_init_method(self):
        """Test the __init__ method of the class defined in the JSON snippet."""
        results = []
        code = self.init_code

        # Compile and extract the class with the __init__ method
        exec_globals = {
            'torch': torch,
            'nn': nn,
            'Rearrange': Rearrange,
            'PerceiverAttention': PerceiverAttention,
            'FeedForward': FeedForward,
            'LayerNorm': LayerNorm
        }
        exec_locals = {}

        try:
            # Execute the code assuming it defines a class with the __init__
            exec(f'class TestClass:\n{code}', exec_globals, exec_locals)
            TestClass = exec_locals['TestClass']

            # Test instantiation
            instance = TestClass(
                dim=128,
                depth=6,
                dim_head=64,
                heads=8,
                num_latents=64,
                num_latents_mean_pooled=4,
                max_seq_len=512,
                ff_mult=4
            )

            # Validate attributes
            self.assertIsInstance(instance.pos_emb, nn.Embedding)
            self.assertEqual(instance.latents.shape, (64, 128))
            self.assertIsNotNone(instance.to_latents_from_mean_pooled_seq)
            self.assertEqual(len(instance.layers), 6)

            results.append({
                "function_name": "__init__",
                "code": code,
                "result": "passed"
            })

        except Exception as e:
            print(f"Initialization failed with error: {e}")
            results.append({
                "function_name": "__init__",
                "code": code,
                "result": "failed"
            })

        # ============= Write test results to test_result.jsonl =============
        # Read existing test_result.jsonl (if exists)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "__init__"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__init__"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()