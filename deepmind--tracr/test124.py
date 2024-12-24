import unittest
import json
import os
from typing import Union

TEST_RESULT_JSONL = "test_result.jsonl"

class SOp:
    pass

class SequenceMap(SOp):
    def __init__(self, func, *args):
        self.func = func
        self.args = args

    def evaluate(self):
        return self.func(*(arg.evaluate() for arg in self.args))

class Map(SOp):
    def __init__(self, func, arg):
        self.func = func
        self.arg = arg

    def evaluate(self):
        return self.func(self.arg.evaluate())

class NumericValue(SOp):
    def __init__(self, value):
        self.value = value

    def evaluate(self):
        return self.value

class TestAndOperator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[123]  # Get the 124th JSON element

    def test_and_operator(self):
        """Test the __and__ method implementation."""
        results = []
        exec_globals = {
            'SOp': SOp,
            'SequenceMap': SequenceMap,
            'Map': Map,
            'NumericValue': NumericValue,
            'Union': Union
        }
        exec_locals = {}

        try:
            # Dynamically execute the __and__ method code
            exec(self.code_snippet, exec_globals, exec_locals)

            # Check if __and__ is in exec_locals
            if '__and__' not in exec_locals:
                print("Code snippet 124: FAILED, '__and__' not found in exec_locals.\n")
                results.append({
                    "function_name": "__and__",
                    "code": self.code_snippet,
                    "result": "failed"
                })
                return

            # Prepare stub objects and perform tests
            and_method = exec_locals['__and__']

            obj1 = NumericValue(True)
            obj2 = NumericValue(False)

            # Test & with another SOp
            sop_result = and_method(obj1, obj2)
            self.assertIsInstance(sop_result, SequenceMap, "Expected a SequenceMap return type.")
            self.assertTrue(sop_result.evaluate())

            # Test & with NumericValue
            num_result = and_method(obj1, True)
            self.assertIsInstance(num_result, Map, "Expected a Map return type.")
            self.assertTrue(num_result.evaluate())

            print("Code snippet 124: PASSED all assertions.\n")
            results.append({
                "function_name": "__and__",
                "code": self.code_snippet,
                "result": "passed"
            })

        except Exception as e:
            print(f"Code snippet 124: FAILED with error: {e}\n")
            results.append({
                "function_name": "__and__",
                "code": self.code_snippet,
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

        # Remove old records with function_name == "__and__"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__and__"
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