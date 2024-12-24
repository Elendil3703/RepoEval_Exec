import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestMakeHist(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[174]  # Get the 175th JSON element

    def test_make_hist(self):
        """Test make_hist function behavior."""

        # Initial validation of code structure
        code = self.code_snippet
        if "def make_hist" not in code:
            self.fail("Function 'make_hist' not found in code snippet.")

        # Define the test input and expected output
        test_input = "abac"
        expected_output = [2, 1, 2, 1]

        # Dynamically execute the code and run the test
        exec_globals = {
            'rasp': MockRasp(),
        }
        exec_locals = {}

        try:
            exec(code, exec_globals, exec_locals)
            if 'make_hist' not in exec_locals:
                self.fail("Function 'make_hist' not executable in provided code.")

            # Extract make_hist function and run the test
            make_hist_func = exec_locals['make_hist']
            hist_instance = make_hist_func()
            actual_output = hist_instance(test_input)

            self.assertEqual(
                actual_output,
                expected_output,
                f"make_hist({test_input!r}) returned {actual_output}, expected {expected_output}.",
            )
            result = {"function_name": "make_hist", "code": code, "result": "passed"}

        except Exception as e:
            result = {
                "function_name": "make_hist",
                "code": code,
                "result": f"failed with error: {e}"
            }

        # Write the test result to JSONL
        self.write_test_result(result)

    @staticmethod
    def write_test_result(result):
        # Read existing records from the JSONL file
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for 'make_hist'
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "make_hist"
        ]

        # Add new result
        existing_records.append(result)

        # Rewrite the JSONL file with updated results
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

class MockRasp:
    """
    Mock class that simulates the relevant portions of the rasp
    library needed for this test. You need to adapt this based
    on the actual behavior of the rasp library.
    """
    class SOp:
        def __init__(self, func):
            self.func = func

        def __call__(self, sequence):
            return self.func(sequence)

    @staticmethod
    def tokens():
        # Mock behavior for RASP tokens function (if necessary)
        pass

    class Comparison:
        EQ = 'eq'

    class Select:
        def __init__(self, a, b, comparison):
            self.a = a
            self.b = b
            self.comparison = comparison

        def named(self, name):
            return self

    class SelectorWidth:
        def __init__(self, select):
            self.select = select

        def named(self, name):
            return TestMakeHist.function_to_mock_selector_width()

    @staticmethod
    def function_to_mock_selector_width():
        """
        Mock method that emulates the behavior of SelectorWidth
        returning a sequence with counts of each token.
        Customize based on your actual requirement.
        """
        def hist_function(sequence):
            from collections import Counter
            counts = Counter(sequence)
            return [counts[token] for token in sequence]
        return MockRasp.SOp(hist_function)

if __name__ == "__main__":
    unittest.main()