import unittest
import json
import os
from typing import Tuple, Union, Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestFindAttrFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[17]  # Get the 17th JSON element (index 16)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 17th JSON array")

    def test_findattr_function(self):
        """Dynamically test all code snippets related to 'findattr'."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # ------------------- Dynamic execution and testing -------------------
                exec_globals = {
                    'Any': Any,
                    'Tuple': Tuple,
                    'Union': Union,
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Ensure 'findattr' is defined
                    if 'findattr' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'findattr' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "findattr",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    findattr = exec_locals['findattr']

                    # Test cases for the function
                    class DummyObject:
                        attr1 = "value1"
                        attr2 = "value2"

                    dummy_obj = DummyObject()

                    # Test 1: Attribute exists
                    try:
                        result = findattr(dummy_obj, ("attr1", "attr3"))
                        self.assertEqual(result, "value1", f"Code snippet {i} failed test 1.")
                    except Exception as e:
                        raise AssertionError(f"Code snippet {i} failed test 1 with error: {e}")

                    # Test 2: Attribute does not exist
                    try:
                        with self.assertRaises(ValueError, msg=f"Code snippet {i} failed test 2."):
                            findattr(dummy_obj, ("attr3", "attr4"))
                    except Exception as e:
                        raise AssertionError(f"Code snippet {i} failed test 2 with error: {e}")

                    # Test 3: Attribute exists but is not first in tuple
                    try:
                        result = findattr(dummy_obj, ("attr3", "attr2"))
                        self.assertEqual(result, "value2", f"Code snippet {i} failed test 3.")
                    except Exception as e:
                        raise AssertionError(f"Code snippet {i} failed test 3 with error: {e}")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "findattr",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "findattr",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for 'findattr'
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "findattr"
        ]

        # Add new results
        existing_records.extend(results)

        # Write updated records back to the file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")


if __name__ == "__main__":
    unittest.main()