import unittest
import json
import os
from typing import Union

TEST_RESULT_JSONL = "test_result.jsonl"

class SOp:
    pass

class SequenceMap:
    def __init__(self, func, other, self_obj):
        self.func = func
        self.other = other
        self.self_obj = self_obj
    
    def execute(self):
        # Imaginary execution logic for testing
        return self.func(True, False)

class Map:
    def __init__(self, func, self_obj):
        self.func = func
        self.self_obj = self_obj
    
    def execute(self):
        # Imaginary execution logic for testing
        return self.func(True)

class TestRandFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[125]  # Get the 126th JSON element (index 125)
        if len(cls.code_snippet) < 1:
            raise ValueError("Expected a valid code snippet in the JSON array")

    def test_rand_function(self):
        """Dynamically test the __rand__ function with additional checks."""
        results = []  # To collect test results for writing to JSONL

        code = self.code_snippet

        try:
            # ------------------- Dynamic execution and testing logic -------------------
            exec_globals = {
                'Union': Union,
                'SOp': SOp,
                'SequenceMap': SequenceMap,
                'Map': Map
            }

            exec(code, exec_globals)

            # Access the dynamically created __rand__ function
            __rand__ = exec_globals['SOp'].__dict__['__rand__']

            # Create instances for testing
            sop_instance = SOp()
            other_instance = SOp()
            numeric_value = 1  # Simulating a NumericValue

            # Test case 1: other is an instance of SOp
            result = __rand__(sop_instance, other_instance)
            if not isinstance(result, SequenceMap) or not result.execute() == False:
                raise AssertionError("__rand__ did not return the expected SequenceMap result with SOp.")

            # Test case 2: other is a numeric value
            result = __rand__(sop_instance, numeric_value)
            if not isinstance(result, Map) or not result.execute() == True:
                raise AssertionError("__rand__ did not return the expected Map result with NumericValue.")

            print("Code snippet: PASSED all assertions.\n")
            results.append({
                "function_name": "__rand__",
                "code": code,
                "result": "passed"
            })

        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}\n")
            results.append({
                "function_name": "__rand__",
                "code": code,
                "result": "failed"
            })

        # ============= Write test results to test_result.jsonl =============
        # Read the existing test_result.jsonl (ignore if not exist)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Delete old records with function_name == "__rand__"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__rand__"
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