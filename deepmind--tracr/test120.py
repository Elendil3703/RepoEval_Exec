import unittest
import json
import sys
import os
from typing import Union

TEST_RESULT_JSONL = "test_result.jsonl"

class SOp:
    pass

class SequenceMap:
    def __init__(self, func, seq1, seq2):
        pass

class Map:
    def __init__(self, func, seq):
        pass

class TestRSubFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[119]  # Get the 120th JSON element (index 119)
        if not cls.code_snippet:
            raise ValueError("Expected code snippet in the 120th JSON array")

    def test_rsub(self):
        """Test the __rsub__ method in the JSON code snippet."""
        passed_count = 0
        failed_count = 0
        results = []

        code = self.code_snippet
        print("Running test for __rsub__ function...")
        
        # Static checks
        if "__rsub__" not in code:
            print("__rsub__ function not found in code.\n")
            failed_count += 1
            results.append({
                "function_name": "__rsub__",
                "code": code,
                "result": "failed"
            })
            return
        
        exec_globals = {
            'Union': Union,
            'SOp': SOp,
            'SequenceMap': SequenceMap,
            'Map': Map
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if __rsub__ is defined
            if '__rsub__' not in exec_locals:
                print("__rsub__ not found in exec_locals.\n")
                failed_count += 1
                results.append({
                    "function_name": "__rsub__",
                    "code": code,
                    "result": "failed"
                })
                return
            
            # Assume further testing of functionality if applies
            # Test cases
            operand_instance = exec_locals['SOp']()
            other_instance = exec_locals['SOp']()
            numeric_value = 10

            rsub_func = exec_locals['__rsub__']

            # Test with other being an instance of SOp
            result = rsub_func(operand_instance, other_instance)
            self.assertIsInstance(
                result,
                SequenceMap,
                "__rsub__ did not return a SequenceMap for SOp instance as other."
            )

            # Test with other being a NumericValue
            result = rsub_func(operand_instance, numeric_value)
            self.assertIsInstance(
                result,
                Map,
                "__rsub__ did not return a Map for numeric value as other."
            )

            print("Code snippet PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "__rsub__",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"Code snippet FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "__rsub__",
                "code": code,
                "result": "failed"
            })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 1\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # Write the test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for function __rsub__
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__rsub__"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()