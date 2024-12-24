import unittest
import json
import sys
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCheckKeys(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[452]  # Get the 453rd JSON element (index 452)
        if not cls.code_snippet:
            raise ValueError("Expected code snippet in the JSON data")

    def test_check_keys(self):
        """Test for the _check_keys function."""
        # Define some example column and against sets for the test
        test_cases = [
            (None, {"a", "b", "c"}, None),  # No columns given, should pass
            ("a", {"a", "b", "c"}, None),  # Valid single column, should pass
            (["a", "b"], {"a", "b", "c"}, None),  # Valid list of columns, should pass
            (["a", "d"], {"a", "b", "c"}, KeyError),  # Invalid keys, should raise KeyError
        ]

        # Results list to write test outcomes
        results = []

        # Preparing globals for exec
        exec_globals = {
            '__builtins__': __builtins__,
        }
        exec_locals = {}

        try:
            # Execute the code snippet from JSON
            exec(self.code_snippet, exec_globals, exec_locals)

            for i, (columns, against, expected_exception) in enumerate(test_cases):
                try:
                    # Test _check_keys function
                    exec_locals['_check_keys'](columns, against)
                    if expected_exception is not None:
                        raise AssertionError("Expected exception not raised")
                    
                    # Record successful run
                    results.append({
                        "function_name": "_check_keys",
                        "code": self.code_snippet,
                        "result": "passed",
                        "test_case": i
                    })
                    
                except Exception as e:
                    if expected_exception is not None and isinstance(e, expected_exception):
                        # Expected exception, record as passed
                        results.append({
                            "function_name": "_check_keys",
                            "code": self.code_snippet,
                            "result": "passed",
                            "test_case": i
                        })
                    else:
                        # Unexpected exception, record as failed
                        results.append({
                            "function_name": "_check_keys",
                            "code": self.code_snippet,
                            "result": "failed",
                            "test_case": i,
                            "error": str(e)
                        })

        except Exception as e:
            # If code execution fails, all tests are considered failed
            results = [{
                "function_name": "_check_keys",
                "code": self.code_snippet,
                "result": "failed",
                "error": str(e)
            }]

        # Write the results to the test_result.jsonl file
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove any existing records for "_check_keys" function
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_check_keys"
        ]

        # Append new results
        existing_records.extend(results)

        # Write to JSONL
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()