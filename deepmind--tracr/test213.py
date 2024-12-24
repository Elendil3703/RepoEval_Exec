import unittest
import json
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class BasisDirection:
    def __init__(self, name, value):
        self.name = name
        self.value = value

class TestBasisDirectionLt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[212]  # Get the 213th JSON element (i.e., index 212)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_lt_method(self):
        """Test the __lt__ method in the code snippets."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                exec_globals = {'BasisDirection': BasisDirection}
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Ensure `__lt__` method is in the exec locals
                    if '__lt__' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, '__lt__' method not found.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "__lt__",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Use the function for testing
                    lt_method = exec_locals['__lt__']

                    # Create test cases to check __lt__
                    obj1 = BasisDirection("a", 10)
                    obj2 = BasisDirection("b", 5)
                    obj3 = BasisDirection("a", 20)
                    obj4 = BasisDirection("a", 10)

                    self.assertTrue(lt_method(obj1, obj2), "lt_method(obj1, obj2) should be True.")
                    self.assertFalse(lt_method(obj2, obj1), "lt_method(obj2, obj1) should be False.")
                    self.assertTrue(lt_method(obj1,obj3), "lt_method(obj1, obj3) should be True.")
                    self.assertFalse(lt_method(obj3, obj4), "lt_method(obj3, obj4) should be False.")
                    self.assertFalse(lt_method(obj1, obj4), "lt_method(obj1, obj4) should be False.")
                    
                    # Test with type error scenario
                    self.assertFalse(lt_method(obj1, "not_a_basis_direction"), "lt_method(obj1, other_type) should be False.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "__lt__",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__lt__",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary information
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write the test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for function_name == "__lt__"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__lt__"
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