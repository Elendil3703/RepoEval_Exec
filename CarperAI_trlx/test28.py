import unittest
import json
import sys
import os
import torch
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithValue

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCarperAITrlxForwardResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[27]  # Get the 29th JSON element (28th index)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in JSON array")

    def test_forward(self):
        """Test the forward function logic from the provided code snippet."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Testing code snippet {i}...")
                exec_globals = {
                    'torch': torch,
                    'Optional': Optional,
                    'List': List,
                    'Tuple': Tuple,
                    'Union': Union,
                    'CausalLMOutputWithValue': CausalLMOutputWithValue,
                }
                exec_locals = {}

                try:
                    exec(code, exec_globals, exec_locals)

                    if 'forward' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'forward' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "forward",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    forward = exec_locals['forward']
                    
                    # Mock objects for inputs and dependencies
                    input_ids = torch.tensor([[1, 2, 3]])
                    attention_mask = torch.tensor([[1, 1, 1]])
                    outputs = torch.rand((1, 3, 768))  # Mocked output

                    class MockModel:
                        def __call__(self, **kwargs):
                            return CausalLMOutputWithValue(
                                logits=outputs,
                                hidden_states=[outputs]
                            )
                    
                    class MockVHead:
                        def __call__(self, h):
                            return torch.mean(h, dim=-1)

                    class MockSelf:
                        def get_compatible_forward_kwargs(self, **kwargs):
                            return kwargs

                        base_model = MockModel()
                        v_head = MockVHead()

                    mock_self = MockSelf()
                    result = forward(
                        mock_self,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                    
                    self.assertIsInstance(result, CausalLMOutputWithValue, 
                                          f"Code snippet {i} did not return CausalLMOutputWithValue")
                    
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

        # Summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records of forward function
        existing_records = [rec for rec in existing_records if rec.get("function_name") != "forward"]

        # Append new results
        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()