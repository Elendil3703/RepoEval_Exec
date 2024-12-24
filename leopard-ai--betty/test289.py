import unittest
import json
import torch
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

def replace_none_with_zero(grads, parameters):
    """Replaces None gradients with zero tensors of the same shape as their corresponding parameters."""
    return [
        torch.zeros_like(param) if grad is None else grad 
        for grad, param in zip(grads, parameters)
    ]

# Sample jvp function mappings
def sample_jvp_fn(jvp, layer_current, layer_next, sync):
    # This is a placeholder for the actual function.
    # In a real scenario, implement the joint vector product logic here.
    return [2 * grad for grad in jvp]

jvp_fn_mapping = {
    "sample_type": sample_jvp_fn
}

class TestGetGrads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[288]  # Get the 289th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array at index 288")

    def test_get_grads(self):
        """Test the behavior of the get_grads function."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                exec_globals = {
                    'torch': torch,
                    'replace_none_with_zero': replace_none_with_zero,
                    'jvp_fn_mapping': jvp_fn_mapping,
                }

                exec_locals = {}

                try:
                    # Dynamic execution of the function
                    exec(code, exec_globals, exec_locals)

                    # Check if get_grads is defined
                    assert 'get_grads' in exec_locals

                    # Set up a mock loss, path, and other required data
                    mock_loss = torch.tensor(1.0, requires_grad=True)
                    mock_parameters = [torch.tensor([1.0], requires_grad=True)]
                    mock_path = [
                        None,
                        type("LayerMock", (), {"trainable_parameters": lambda s: mock_parameters})(),
                        type("LayerConfigMock", (), {})(),
                        type("LayerMock", (), {"config": type("Config", (), {"type": "sample_type"})})(),
                    ]

                    # Adjust path to handle any required logic for the mock
                    mock_path[1].config = mock_path[2].config
                    mock_do_sync = False

                    # Call the function to test
                    grads = exec_locals['get_grads'](mock_loss, mock_path, True, mock_do_sync)

                    # Perform assertions
                    for grad in grads:
                        self.assertTrue(torch.equal(grad, torch.zeros_like(mock_parameters[0]) * 2))

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "get_grads",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "get_grads",
                        "code": code,
                        "result": "failed"
                    })

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

        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "get_grads"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()