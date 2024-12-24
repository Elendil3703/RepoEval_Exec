import unittest
import json
import os
from typing import List, Dict, Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestDecodeFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[247]  # Get the 248th JSON element (index 247)
        if not cls.code_snippet:
            raise ValueError("Expected a code snippet in the JSON data element")

    def test_decode_function(self):
        """Test the decode function from the code snippet."""
        passed_count = 0
        failed_count = 0
        results = []

        # Dynamically execute the code snippet to retrieve the class definition
        exec_globals = {
            'List': List,
            'Any': Any,
        }
        exec_locals = {}

        try:
            exec(self.code_snippet, exec_globals, exec_locals)
            decode_class = None

            for name, obj in exec_locals.items():
                if hasattr(obj, 'decode') and callable(getattr(obj, 'decode')):
                    decode_class = obj
                    break

            if decode_class is None:
                raise ValueError("No class found with a 'decode' method in the code snippet.")
            
            # Now we have the class, test its decode method
            encoding_map = {1: 'a', 2: 'b', 3: 'c'}
            decoding_class_instance = decode_class()
            setattr(decoding_class_instance, 'encoding_map', encoding_map)

            test_cases = [
                ([1, 2, 3], ['a', 'b', 'c']),
                ([3, 2, 1], ['c', 'b', 'a']),
                ([1, 1, 2], ['a', 'a', 'b']),
                ([2], ['b']),
            ]

            for i, (inputs, expected) in enumerate(test_cases):
                with self.subTest(test_case=i):
                    decoded_result = decoding_class_instance.decode(inputs)
                    self.assertEqual(decoded_result, expected)
                    passed_count += 1
                    results.append({
                        "function_name": "decode",
                        "code": self.code_snippet,
                        "result": "passed"
                    })
            
            # Test for exception when input contains invalid tokens
            invalid_test_cases = [
                ([4], "Inputs {4} not found in decoding map"),
            ]

            for i, (inputs, expected_exception) in enumerate(invalid_test_cases):
                with self.subTest(invalid_case=i):
                    with self.assertRaises(ValueError) as context:
                        decoding_class_instance.decode(inputs)
                    self.assertIn(expected_exception, str(context.exception))
                    passed_count += 1
                    results.append({
                        "function_name": "decode",
                        "code": self.code_snippet,
                        "result": "passed"
                    })

        except Exception as e:
            print(f"Code snippet failed with error: {e}")
            failed_count += 1
            results.append({
                "function_name": "decode",
                "code": self.code_snippet,
                "result": "failed"
            })

        # Final summary
        print(f"Test Summary: {passed_count} passed, {failed_count} failed")

        # Write test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for "decode"
        existing_records = [
            rec for rec in existing_records if rec.get("function_name") != "decode"
        ]

        # Append new results
        existing_records.extend(results)

        # Write updated records
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()