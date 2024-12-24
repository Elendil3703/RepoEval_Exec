import unittest
import json
import os
from typing import Union  # Import necessary types

TEST_RESULT_JSONL = "test_result.jsonl"

# Mock implementations of the classes used in the function
class VectorInBasis:
    def __init__(self, basis_directions):
        self.basis_directions = basis_directions

class BasisDirection:
    pass

class TestContainsMethod(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file and select the 220th (index 219) code snippet data
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[219]
        if not cls.code_snippet:
            raise ValueError("Expected a valid code snippet at index 219 in the JSON data")

    def test_contains_method(self):
        """Testing the __contains__ method in the code snippet."""
        passed_count = 0
        failed_count = 0
        results = []

        code = self.code_snippet
        exec_globals = {
            'Union': Union,
            'VectorInBasis': VectorInBasis,
            'BasisDirection': BasisDirection,
        }
        exec_locals = {}

        try:
            # Execute the provided code snippet
            exec(code, exec_globals, exec_locals)

            # Check if __contains__ is in executed locals
            if '__contains__' not in exec_locals:
                raise ValueError("'__contains__' method not found in the executed code.")

            contains_method = exec_locals['__contains__']

            # Mock object implementing a basis for testing
            class MockObject:
                def __init__(self, basis):
                    self.basis = basis

            # Sample test cases
            mock_basis = [BasisDirection(), BasisDirection()]

            # Test case 1: Item is a BasisDirection and is in basis
            item1 = mock_basis[0]
            basis_obj1 = MockObject(mock_basis)
            result1 = contains_method(basis_obj1, item1)
            self.assertTrue(result1, "Expected item to be in basis")

            # Test case 2: Item is a VectorInBasis with same basis directions
            item2 = VectorInBasis(mock_basis)
            basis_obj2 = MockObject(mock_basis)
            result2 = contains_method(basis_obj2, item2)
            self.assertTrue(result2, "Expected VectorInBasis with same directions to be equal")

            # Test case 3: Item is a VectorInBasis with different basis directions
            different_directions = [BasisDirection()]
            item3 = VectorInBasis(different_directions)
            basis_obj3 = MockObject(mock_basis)
            result3 = contains_method(basis_obj3, item3)
            self.assertFalse(result3, "Expected VectorInBasis with different directions to be unequal")

            passed_count += 3
            results.append({
                "function_name": "__contains__",
                "code": code,
                "result": "passed"
            })
        except AssertionError as e:
            failed_count += 1
            results.append({
                "function_name": "__contains__",
                "code": code,
                "result": "failed"
            })
            print(f"Test FAILED with AssertionError: {e}")
        except Exception as e:
            failed_count += 1
            results.append({
                "function_name": "__contains__",
                "code": code,
                "result": "failed"
            })
            print(f"Test FAILED with error: {e}")

        # Summary and results writing for __contains__
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 3\n")
        self.assertEqual(passed_count + failed_count, 3, "Test count mismatch!")

        # Write test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Delete old records for __contains__
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__contains__"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()