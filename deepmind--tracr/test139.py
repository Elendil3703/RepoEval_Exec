import unittest
import json
import os
from typing import Any  

TEST_RESULT_JSONL = "test_result.jsonl"

class TestDefaultEncodingResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file and select the code snippet at index 138
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[138]  # Get the 139th JSON element

        if not cls.code_snippet:
            raise ValueError("Expected a code snippet in the selected JSON element")

    def test_default_encoding(self):
        """Test the default_encoding function for expected behavior."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        # Initialize the dynamic execution context
        exec_globals = {
            'SOp': type('SOp', (object,), {}),  # Mock SOp type
            'Encoding': type('Encoding', (object,), {'CATEGORICAL': 'CATEGORICAL'}),  # Mock Encoding
            'Any': Any,
        }
        exec_locals = {}

        # Run the code snippet dynamically
        try:
            exec(self.code_snippet, exec_globals, exec_locals)

            # Check if default_encoding is correctly defined
            self.assertIn('default_encoding', exec_locals, "Function 'default_encoding' not found.")
            default_encoding = exec_locals['default_encoding']
            
            # Test case: valid input (instance of SOp)
            try:
                sop_instance = exec_globals['SOp']()
                result = default_encoding(sop_instance)
                self.assertEqual(result, exec_globals['Encoding'].CATEGORICAL)
                print("Test case: valid input (instance of SOp) - PASSED")
                passed_count += 1
                results.append({
                    "function_name": "default_encoding",
                    "code": self.code_snippet,
                    "result": "passed",
                    "test_case": "valid_input_instance_sop"
                })
            except Exception as e:
                print(f"Test case: valid input (instance of SOp) - FAILED with error {e}")
                failed_count += 1
                results.append({
                    "function_name": "default_encoding",
                    "code": self.code_snippet,
                    "result": "failed",
                    "test_case": "valid_input_instance_sop"
                })

            # Test case: invalid input (not an instance of SOp)
            try:
                with self.assertRaises(TypeError):
                    default_encoding("not_an_sop_instance")
                print("Test case: invalid input (not an instance of SOp) - PASSED")
                passed_count += 1
                results.append({
                    "function_name": "default_encoding",
                    "code": self.code_snippet,
                    "result": "passed",
                    "test_case": "invalid_input_not_sop_instance"
                })
            except Exception as e:
                print(f"Test case: invalid input (not an instance of SOp) - FAILED with error {e}")
                failed_count += 1
                results.append({
                    "function_name": "default_encoding",
                    "code": self.code_snippet,
                    "result": "failed",
                    "test_case": "invalid_input_not_sop_instance"
                })

        except Exception as exec_error:
            print(f"Dynamic execution of code snippet failed with error: {exec_error}")
            failed_count += 1
            results.append({
                "function_name": "default_encoding",
                "code": self.code_snippet,
                "result": "failed"
            })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed")

        # Write the test results to the test_result.jsonl file
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Filter out old records of the same function
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "default_encoding"
        ]

        # Append new results
        existing_records.extend(results)

        # Write the updated records to the JSONL file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()