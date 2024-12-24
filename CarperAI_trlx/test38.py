import unittest
import json
import torch
import os
from typing import List, Tuple

TEST_RESULT_JSONL = "test_result.jsonl"

class TestForwardMethod(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[37]  # Get the 38th JSON element (index 37)
        if not cls.code_snippet:
            raise ValueError("Expected code snippet in the 38th JSON array")

    def test_forward(self):
        """Dynamically test the forward function logic."""

        results = []  # Collect results for writing to JSONL

        # Execute the code snippet assuming it defines necessary components
        exec_globals = {
            'torch': torch,
        }
        exec_locals = {}

        try:
            # Dynamic execution of the code
            exec(self.code_snippet, exec_globals, exec_locals)

            # Ensure forward function exists and test its output
            if 'forward' not in exec_locals:
                self.fail("The code snippet did not define a 'forward' function.")

            forward_function = exec_locals['forward']

            # Assuming 'forward' is an unbound function, mock its environment
            class MockModel:
                def __init__(self):
                    # Mock q_heads and target_q_heads with simple functions
                    self.q_heads = [lambda x: x + 1 for _ in range(2)]
                    self.target_q_heads = [lambda x: x + 2 for _ in range(2)]
                    self.v_head = lambda x: x + 3

            model = MockModel()

            def batched_index_select(tensor, indices, dim):
                # Simple batched index select mocking
                return tensor.index_select(dim, indices)

            # Test setup
            hs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            states_ixs = torch.tensor([0, 1])
            actions_ixs = torch.tensor([1, 0])

            # Call the forward function
            qs, target_qs, vs = forward_function(model, hs, states_ixs, actions_ixs)

            # Assertions to check expected outcomes
            self.assertEqual(len(qs), len(model.q_heads), "Mismatch in number of q_heads outputs")
            self.assertEqual(len(target_qs), len(model.target_q_heads), "Mismatch in number of target_q_heads outputs")
            self.assertTrue(torch.equal(vs, model.v_head(hs.index_select(1, states_ixs))), "Incorrect v_head output")

            results.append({
                "function_name": "forward",
                "code": self.code_snippet,
                "result": "passed"
            })

            print("Test for forward function: PASSED all assertions.")
        except Exception as e:
            print(f"Test for forward function: FAILED with error: {e}")
            results.append({
                "function_name": "forward",
                "code": self.code_snippet,
                "result": "failed"
            })

        # Write the results to the test_result.jsonl file
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records related to the forward function
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

if __name__ == "__main__":
    unittest.main()