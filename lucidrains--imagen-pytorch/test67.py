import unittest
import json
import os
import torch
from torch import nn
from einops import repeat

TEST_RESULT_JSONL = "test_result.jsonl"

# A utility function to check if an object exists (similar to your "exists" function)
def exists(x):
    return x is not None

class MockAttentionLayer(nn.Module):
    def forward(self, x_with_pos, latents, mask=None):
        # Simulates an attention operation for testing purposes
        return latents * 0.5

class MockFeedForwardLayer(nn.Module):
    def forward(self, latents):
        # Simulates a feedforward operation for testing purposes
        return latents + 0.1

class MockEncoder(nn.Module):
    def __init__(self, pos_emb, latents, layers, to_latents_from_mean_pooled_seq=None):
        super().__init__()
        self.pos_emb = pos_emb
        self.latents = latents
        self.layers = layers
        self.to_latents_from_mean_pooled_seq = to_latents_from_mean_pooled_seq

    def forward(self, x, mask=None):
        n, device = x.shape[1], x.device
        pos_emb = self.pos_emb(torch.arange(n, device = device))

        x_with_pos = x + pos_emb

        latents = repeat(self.latents, 'n d -> b n d', b = x.shape[0])

        if exists(self.to_latents_from_mean_pooled_seq):
            meanpooled_seq = torch.mean(x, dim=1)
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim=-2)

        for attn, ff in self.layers:
            latents = attn(x_with_pos, latents, mask=mask) + latents
            latents = ff(latents) + latents

        return latents

class TestForward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[66]  # Get the 67th JSON element

    def test_forward(self):
        """Test the forward method of the mocked encoder."""
        passed_count = 0
        failed_count = 0
        results = []  # Collect test results

        with self.subTest():
            print("Running test for the forward function...")
            try:
                # Mocking the necessary components
                pos_emb = nn.Embedding(10, 16)  # Mock positional embedding
                latents = torch.randn(10, 16)   # Mock latents
                layers = [(MockAttentionLayer(), MockFeedForwardLayer()) for _ in range(2)]

                encoder = MockEncoder(pos_emb, latents, layers)

                # Run the forward function with random input
                x = torch.randn(4, 10, 16)  # Example input tensor
                output = encoder.forward(x)

                # Check output shape
                self.assertEqual(output.shape, (4, 10, 16), "Output shape mismatch.")

                # Check if latents updated as expected
                expected_output = latents * 0.5 + latents + 0.1
                self.assertTrue(torch.allclose(output, expected_output, atol=1e-5), "Output values mismatch.")

                print("forward function: PASSED all assertions.\n")
                passed_count += 1
                results.append({
                    "function_name": "forward",
                    "code": self.code_snippet,
                    "result": "passed"
                })
            except Exception as e:
                print(f"forward function: FAILED with error: {e}\n")
                failed_count += 1
                results.append({
                    "function_name": "forward",
                    "code": self.code_snippet,
                    "result": "failed"
                })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed\n")
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

        # Remove old records for the same function
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "forward"
        ]

        # Append new results
        existing_records.extend(results)

        # Re-write the JSONL file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()