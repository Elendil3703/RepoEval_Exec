import unittest
import json
import os
from typing import List, Any  # Ensure these are available in the injected environment

TEST_RESULT_JSONL = "test_result.jsonl"

class TestEncodeFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[246]  # Get the 247th JSON element (index 246)
        if not cls.code_snippet:
            raise ValueError("Expected a code snippet at index 246 in the JSON array")

    def test_encode_function(self):
        """Dynamically test the encode function from the JSON with specific cases."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to the JSONL file

        code = self.code_snippet
        exec_globals = {'List': List, 'Any': Any}
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if the encode function really exists
            if 'encode' not in exec_locals:
                self.fail("The encode function was not found after executing the provided code snippet.")
                results.append({
                    "function_name": "encode",
                    "code": code,
                    "result": "failed"
                })
                return

            # Create an instance with required attributes based on the assumptions from the function code
            class MockBasesValue:
                def __init__(self, value):
                    self.value = value

                def __eq__(self, other):
                    return self.value == other

            # Mock class with encode function
            encode = exec_locals['encode']
            mocked_instance = type('MockedInstance', (), {
                'enforce_bos': True,
                'bos_token': MockBasesValue('BOS'),
                'encoding_map': {'BOS': 0, 'A': 1, 'B': 2, 'C': 3},
                '_max_seq_len': 5
            })

            # Tests:
            # 1. Correct encoding
            mocked_obj = mocked_instance()
            inputs = [MockBasesValue('BOS'), MockBasesValue('A'), MockBasesValue('B')]
            expected_output = [0, 1, 2]
            output = encode(mocked_obj, inputs)
            self.assertEqual(output, expected_output, "Encoding did not produce expected result.")
            passed_count += 1
            results.append({
                "function_name": "encode",
                "code": code,
                "result": "passed"
            })

            # 2. Raise error on missing BOS token
            with self.assertRaises(ValueError) as cm:
                encode(mocked_obj, [MockBasesValue('A'), MockBasesValue('B')])
            self.assertIn("First input token must be BOS token", str(cm.exception))
            passed_count += 1
            results.append({
                "function_name": "encode",
                "code": code,
                "result": "passed"
            })

            # 3. Raise error on unknown token
            with self.assertRaises(ValueError) as cm:
                encode(mocked_obj, [MockBasesValue('BOS'), MockBasesValue('D')])
            self.assertIn("not found in encoding", str(cm.exception))
            passed_count += 1
            results.append({
                "function_name": "encode",
                "code": code,
                "result": "passed"
            })

            # 4. Raise error when sequence is too long
            with self.assertRaises(ValueError) as cm:
                encode(mocked_obj, [MockBasesValue('BOS')] * 6)
            self.assertIn("are longer than the maximum sequence length", str(cm.exception))
            passed_count += 1
            results.append({
                "function_name": "encode",
                "code": code,
                "result": "passed"
            })

        except Exception as e:
            print(f"Code snippet execution failed with error: {e}")
            failed_count += 1
            results.append({
                "function_name": "encode",
                "code": code,
                "result": "failed"
            })

        # Summary statistics
        print(f"Test Summary: {passed_count} passed, {failed_count} failed, for 4 tests")

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for the function_name == "encode"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "encode"
        ]

        # Append new results
        existing_records.extend(results)

        # Write new results to the file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")