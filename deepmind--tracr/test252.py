import unittest
import json
import sys
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCarperAITrlxResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[251]  # Get the 252nd JSON element

    def test_code_snippet(self):
        """Test the forward function in the code snippet with synthetic mocks."""
        passed_count = 0
        failed_count = 0
        results = []

        code = self.code_snippet
        print("Running test for the forward function...")

        # Check if 'Transformer' and 'TransformerConfig' are in the snippet
        if 'Transformer' not in code or 'TransformerConfig' not in code:
            print("Code snippet: FAILED, 'Transformer' or 'TransformerConfig' not found in code.\n")
            failed_count += 1
            # Write failure record
            results.append({
                "function_name": "forward",
                "code": code,
                "result": "failed"
            })
        else:
            exec_globals = {
                'model': Any,  # Mock model context with Any
                'causal': True,  # Mock the causal variable
            }
            exec_locals = {}

            try:
                # Dynamically execute the code snippet
                exec(code, exec_globals, exec_locals)

                # Check if 'forward' function exists
                if 'forward' not in exec_locals:
                    print("Code snippet: FAILED, 'forward' function not found after execution.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "forward",
                        "code": code,
                        "result": "failed"
                    })
                else:
                    # Use the forward function and mock its inputs
                    forward = exec_locals['forward']

                    # Mock inputs
                    emb_mock = "mock_emb"
                    mask_mock = "mock_mask"

                    # Try calling the forward function
                    try:
                        result = forward(emb_mock, mask_mock)
                        # Add more assertions based on expected behavior of the forward function
                        # Here we're just checking 'result' exists for demonstrative purposes
                        self.assertIsNotNone(result, "The forward function did not return a result.")
                        print("Code snippet: PASSED all assertions.\n")
                        passed_count += 1
                        results.append({
                            "function_name": "forward",
                            "code": code,
                            "result": "passed"
                        })
                    except Exception as ex:
                        print(f"Code snippet: FAILED during forward execution, error: {ex}\n")
                        failed_count += 1
                        results.append({
                            "function_name": "forward",
                            "code": code,
                            "result": "failed"
                        })
            except Exception as e:
                print(f"Code snippet: FAILED with error: {e}\n")
                failed_count += 1
                results.append({
                    "function_name": "forward",
                    "code": code,
                    "result": "failed"
                })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 1\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # Write the results to test_result.jsonl
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
            if rec.get("function_name") != "forward"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()