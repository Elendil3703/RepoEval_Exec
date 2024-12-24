import unittest
import json
from collections.abc import Mapping, Sequence
from typing import Dict
import copy
import torch

TEST_RESULT_JSONL = "test_result.jsonl"

class TestForwardFunction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[401]  # Get the 402nd JSON element (index 401)
        if not cls.code_snippet:
            raise ValueError("Expected a code snippet at the 402nd position in the JSON array")

    def test_forward_function(self):
        """Dynamically test the forward function in the code snippet."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippet):
            with self.subTest(code_index=i):
                exec_globals = {
                    'copy': copy,
                    'torch': torch,
                    'Mapping': Mapping,
                    'Sequence': Sequence,
                }
                exec_locals = {}

                try:
                    exec(code, exec_globals, exec_locals)

                    forward_function = exec_locals.get('forward')
                    if forward_function is None:
                        raise AssertionError("Function 'forward' not found.")

                    # Mock class to test the forward method
                    class MockClass:
                        def __init__(self):
                            self.outputs = {}
                            self.input_key = None
                            self.handle_list_inputs = True
                        
                        def forward_sub_batch(self, sub_batch, *args, **kwargs):
                            self.outputs[sub_batch['key']] = sub_batch['vision']
                    
                    instance = MockClass()

                    # Test cases
                    test_batches = [
                        ({"key": {"vision": torch.tensor([1, 2, 3])}}, {"key": torch.tensor([1, 2, 3])}),
                        ({"key": {"vision": [torch.tensor([1]), torch.tensor([2])]}}, {"key": torch.tensor([1, 2])}),
                    ]

                    for batch, expected_output in test_batches:
                        result = forward_function(instance, batch)
                        self.assertEqual(result, expected_output, f"Test failed for batch: {batch}")

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

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippet)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippet), "Test count mismatch!")

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in results:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()