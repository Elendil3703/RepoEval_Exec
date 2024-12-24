import unittest
import json
import os
from typing import Any
from torch import einsum
from einops import rearrange
import torch

TEST_RESULT_JSONL = "test_result.jsonl"

class MockNet:
    def __call__(self, x):
        return x

class ForwardClass:
    def __init__(self):
        self.to_k = lambda x: x
        self.net = MockNet()
    
    def forward(self, x):
        context = self.to_k(x)
        x, context = rearrange_many((x, context), 'b n ... -> b n (...)')
        out = einsum('b i n, b c n -> b c i', context.softmax(dim=-1), x)
        out = rearrange(out, '... -> ... 1')
        return self.net(out)

def rearrange_many(tensors, pattern):
    return [rearrange(t, pattern) for t in tensors]

class TestForwardFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[79]  # Get the 80th JSON element

    def test_forward_snippets(self):
        """Dynamically test all forward code snippets in the JSON."""
        passed_count = 0
        failed_count = 0
        results = []

        expected_output_shape = (2, 3, 1)

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                exec_globals = {
                    'torch': torch,
                    'einsum': einsum,
                    'rearrange': rearrange,
                    'rearrange_many': rearrange_many,
                    'MockNet': MockNet,
                }
                exec_locals = {}

                try:
                    # Dynamic execution of the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Create a mock input tensor
                    input_tensor = torch.randn(2, 3, 4)

                    # Get the forward function from the executed code's class instance
                    cls_instance = exec_locals['ForwardClass']()
                    output = cls_instance.forward(input_tensor)

                    # Check the shape of the output
                    self.assertEqual(
                        output.shape,
                        expected_output_shape,
                        f"Code snippet {i} forward output shape mismatch."
                    )

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

        # Overall Test Summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Writing results to the test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for the function 'forward'
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

if __name__ == "__main__":
    unittest.main()