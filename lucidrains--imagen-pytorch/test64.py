import unittest
import json
import os
import torch

TEST_RESULT_JSONL = "test_result.jsonl"

class TestForwardFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[63]  # Get the 64th JSON element (index 63)
        if not cls.code_snippet:
            raise ValueError("Expected a code snippet at index 63")

    def test_forward(self):
        """Dynamically test the forward function in the JSON snippet."""
        results = []  # Collect results to write to JSONL

        # Setup environment for executing the code snippet
        exec_globals = {'torch': torch}
        exec_locals = {}

        try:
            # Execute the code snippet
            exec(self.code_snippet, exec_globals, exec_locals)

            # Check if the `forward` function exists
            if 'forward' not in exec_locals:
                print("FAILED, 'forward' function not found after execution.")
                results.append({
                    "function_name": "forward",
                    "code": self.code_snippet,
                    "result": "failed"
                })
                return

            forward = exec_locals['forward']

            class MockModule:
                def __init__(self, dim, stable, g):
                    self.dim = dim
                    self.stable = stable
                    self.g = g

            # Define test cases
            test_cases = [
                (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), 1, True),
                (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), 0, False),
                (torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64), 1, True),
            ]

            passed_count = 0
            failed_count = 0

            # Run test cases
            for i, (input_tensor, dim, stable) in enumerate(test_cases):
                with self.subTest(test_index=i):
                    mock_module = MockModule(dim, stable, torch.tensor([1.0, 1.0]))

                    try:
                        result = forward(mock_module, input_tensor)
                        self.assertEqual(result.shape, input_tensor.shape, "Output shape mismatch.")
                        passed_count += 1
                        results.append({
                            "function_name": "forward",
                            "test_index": i,
                            "result": "passed"
                        })
                    except Exception as e:
                        failed_count += 1
                        results.append({
                            "function_name": "forward",
                            "test_index": i,
                            "result": "failed",
                            "error": str(e)
                        })

            print(f"\nTest Summary: {passed_count} passed, {failed_count} failed\n")

            # Assert all test cases ran
            self.assertEqual(passed_count + failed_count, len(test_cases), "Test count mismatch!")

        except Exception as e:
            print(f"FAILED with error: {e}")
            results.append({
                "function_name": "forward",
                "code": self.code_snippet,
                "result": "failed",
                "error": str(e)
            })

        # ============= Write Test Results to test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        existing_records.append(json.loads(line))

        # Remove old records for 'forward'
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "forward"
        ]

        # Append new results
        existing_records.extend(results)

        # Write to file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()