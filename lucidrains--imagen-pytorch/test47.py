import unittest
import json
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class TestFirstFunctionResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[46]  # Getting the 47th code (index 46)

        if len(cls.code_snippet) == 0:
            raise ValueError("Expected code snippet in the 47th JSON array element")

    def test_first_function(self):
        """Test the 'first' function logic in the given code snippet."""
        passed_count = 0
        failed_count = 0
        results = []

        code = self.code_snippet

        # Creating a temporary local environment to execute
        exec_globals = {}
        exec_locals = {}

        try:
            # Execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if 'first' function exists
            if 'first' not in exec_locals:
                raise ValueError("'first' function not found in the provided code snippet.")

            # Reference to the first function
            first = exec_locals['first']

            # Define test cases for the function
            test_cases = [
                ([], None, None),  # Test empty array with default
                ([], "default", "default"),  # Test empty array with a given default
                ([1, 2, 3], None, 1),  # Test non-empty array
                (["a", "b", "c"], None, "a"),  # Test array with strings
                ([None], "something", None)  # Test array with None
            ]

            for idx, (arr, default, expected) in enumerate(test_cases):
                with self.subTest(test_case=idx):
                    result = first(arr, default)
                    self.assertEqual(result, expected, f"Failed for input: arr={arr}, d={default}")
                    passed_count += 1
                    results.append({
                        "function_name": "first",
                        "input": f"arr={arr}, d={default}",
                        "result": "passed"
                    })

            print(f"All tests passed for 'first' function.")

        except Exception as e:
            print(f"Function test failed with error: {e}")
            failed_count += 1
            results.append({
                "function_name": "first",
                "code": code,
                "result": "failed"
            })

        # Verify all test cases were executed
        self.assertEqual(passed_count + failed_count, len(test_cases), "Test count mismatch!")

        # ============= Write the test results to test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for function_name == "first"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "first"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()