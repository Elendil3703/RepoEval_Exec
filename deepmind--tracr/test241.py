import unittest
import json
import sys
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestForwardSuperposition(unittest.TestCase):
    def setUp(self):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        # Get the 241st JSON element (index 240)
        self.code_snippet = data[240]
        if not self.code_snippet:
            raise ValueError("Expected code snippet at index 240")

    def test_forward_superposition(self):
        """Dynamically test the forward_superposition function."""
        passed_count = 0
        failed_count = 0
        results = []

        code = self.code_snippet
        # Static Check: Ensure function definition exists
        if "def forward_superposition" not in code:
            print(f"Code snippet: FAILED, function 'forward_superposition' not found.\n")
            failed_count += 1
            results.append({
                "function_name": "forward_superposition",
                "code": code,
                "result": "failed"
            })
            return

        # Dynamic execution
        exec_globals = {
            'compressed_model': type('CompressedModel', (), {'CompressedTransformer': lambda config: type('MockTransformer', (), {'output': 'mocked_output'})()}),
            'model_size': 512,
            'unembed_at_every_layer': False,
            'config': {}
        }
        exec_locals = {}

        try:
            # Execute the code snippet
            exec(code, exec_globals, exec_locals)
            
            # Check if forward_superposition was defined
            if 'forward_superposition' not in exec_locals:
                print(f"Code snippet: FAILED, 'forward_superposition' not found in exec_locals.\n")
                failed_count += 1
                results.append({
                    "function_name": "forward_superposition",
                    "code": code,
                    "result": "failed"
                })
                return

            # Test the forward_superposition function
            forward_superposition = exec_locals['forward_superposition']
            result = forward_superposition('mock_emb', 'mock_mask')

            self.assertEqual(result, 'mocked_output', "The function did not return the expected output.")
            print(f"Code snippet: PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "forward_superposition",
                "code": code,
                "result": "passed"
            })
            
        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "forward_superposition",
                "code": code,
                "result": "failed"
            })

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
            if rec.get("function_name") != "forward_superposition"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()