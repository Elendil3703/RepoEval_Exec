import unittest
import json
import os
import sys
import transformers
from unittest.mock import MagicMock

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCarperAITrlxInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[38]  # Get the 39th JSON element (index 38)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 39th JSON array")

    def test_init_function(self):
        """Test the '__init__' function logic in the code snippets."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Dynamically create a mock of the required components
                base_model_mock = MagicMock(spec=transformers.PreTrainedModel)
                config_mock = MagicMock()
                config_mock.vocab_size = 30522
                base_model_mock.config = config_mock

                lm_head_mock = MagicMock()
                lm_head_mock.parameters.return_value = [MagicMock(dtype='float32')]

                exec_globals = {
                    'transformers': transformers,
                    'hf_get_hidden_size': lambda config: 768,
                    'hf_get_lm_head': lambda model: lm_head_mock,
                    'ILQLHeads': MagicMock(),
                }
                exec_locals = {}

                try:
                    # Execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Create an instance of the class using the provided __init__
                    instance = exec_locals['YourClassName'](base_model=base_model_mock, two_qs=False, alpha=0.95)

                    # Assert the internal state
                    self.assertEqual(instance.two_qs, False, f"Code snippet {i} did not set `two_qs` correctly.")
                    self.assertEqual(instance.alpha, 0.95, f"Code snippet {i} did not set `alpha` correctly.")
                    exec_globals['ILQLHeads'].assert_called_once_with(
                        768, 30522, False, 0.95, dtype='float32'
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "passed"
                    })

                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })

        # Final test summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for the function '__init__'
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__init__"
        ]

        # Append new results
        existing_records.extend(results)

        # Write back to test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()