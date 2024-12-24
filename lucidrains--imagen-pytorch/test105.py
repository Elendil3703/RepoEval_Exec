import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestUnetNumberValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[104]  # Get the 105th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 105th JSON array")

    def test_code_snippets(self):
        """Dynamically test the validate_unet_number function."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                if "def validate_unet_number" not in code:
                    print(f"Code snippet {i}: FAILED, function 'validate_unet_number' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "validate_unet_number",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'Any': Any,
                    'default': lambda x, y: x if x is not None else y,  # Mock default function
                }
                exec_locals = {}

                try:
                    exec(code, exec_globals, exec_locals)

                    # Define a mock class with num_unets attribute and validate_unet_number method
                    class MockModel:
                        def __init__(self, num_unets):
                            self.num_unets = num_unets

                        validate_unet_number = exec_locals['validate_unet_number']

                    # Test scenarios
                    test_model = MockModel(num_unets=1)

                    # Test case 1: When num_unets is 1 and unet_number is None, should default to 1
                    result = test_model.validate_unet_number(None)
                    self.assertEqual(result, 1, f"Code snippet {i} did not handle unet_number=None correctly.")

                    # Test case 2: Correct number within range
                    test_model.num_unets = 3
                    result = test_model.validate_unet_number(2)
                    self.assertEqual(result, 2, f"Code snippet {i} failed for valid unet_number.")

                    # Test case 3: Invalid unet_number
                    with self.assertRaises(AssertionError, msg=f"Code snippet {i} did not raise assertion for invalid unet_number."):
                        test_model.validate_unet_number(0)

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "validate_unet_number",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "validate_unet_number",
                        "code": code,
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

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
            if rec.get("function_name") != "validate_unet_number"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()