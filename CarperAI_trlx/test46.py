import unittest
import json
import os
import sys
from unittest.mock import MagicMock

TEST_RESULT_JSONL = "test_result.jsonl"

class TestSavePretrainedFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[45]  # Get the 46th JSON element (index 45)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_save_pretrained_snippets(self):
        """Dynamically test all code snippets for 'save_pretrained'."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for save_pretrained code snippet {i}...")

                exec_globals = {
                    'sys': sys,
                }
                exec_locals = {}

                # Setup a mock class to test save_pretrained
                class MockModel:
                    def __init__(self):
                        self.base_model = MagicMock()
                        self.state_dict = MagicMock(return_value={'mock_key': 'mock_value'})

                    def save_pretrained(self, *args, **kwargs):
                        state_dict = kwargs.pop("state_dict", None)
                        if state_dict is None:
                            state_dict = self.state_dict()
                            kwargs["state_dict"] = state_dict
                        return self.base_model.save_pretrained(*args, **kwargs)

                exec_locals['MockModel'] = MockModel

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Initialize the MockModel and call save_pretrained
                    mock_model_instance = exec_locals['MockModel']()
                    
                    # Call the save_pretrained method
                    mock_model_instance.save_pretrained('./saved_model', custom_arg='custom_value')

                    # Assertions
                    mock_model_instance.base_model.save_pretrained.assert_called_once()
                    args, kwargs = mock_model_instance.base_model.save_pretrained.call_args
                    self.assertIn('state_dict', kwargs, f"Code snippet {i}: state_dict not passed in kwargs")
                    self.assertIsNotNone(
                        kwargs['state_dict'],
                        f"Code snippet {i}: state_dict is None"
                    )
                    self.assertEqual(
                        kwargs['state_dict'],
                        mock_model_instance.state_dict(),
                        f"Code snippet {i}: state_dict doesn't match method output"
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "save_pretrained",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "save_pretrained",
                        "code": code,
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write results to test_result.jsonl =============
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
            if rec.get("function_name") != "save_pretrained"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()