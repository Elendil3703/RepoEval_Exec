import unittest
import json
import os
import dataclasses
from typing import Any, Sequence

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCreateListOfAttributesToCompare(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[356]  # Get the 357th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 357th JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                if "_create_list_of_attributes_to_compare" not in code:
                    print(f"Code snippet {i}: FAILED, '_create_list_of_attributes_to_compare' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_create_list_of_attributes_to_compare",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Dynamic execution
                exec_globals = {'dataclasses': dataclasses, 'Any': Any, 'Sequence': Sequence}
                exec_locals = {}

                try:
                    exec(code, exec_globals, exec_locals)

                    if '_create_list_of_attributes_to_compare' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, '_create_list_of_attributes_to_compare' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "_create_list_of_attributes_to_compare",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test functionality
                    @dataclasses.dataclass
                    class MockClass:
                        a: int = dataclasses.field(compare=True)
                        b: int = dataclasses.field(compare=True)
                        c: int = dataclasses.field(compare=False)
                    
                    mmm_instance = MockClass(a=1, b=2)
                    attributes_list = exec_locals['_create_list_of_attributes_to_compare'](mmm_instance)

                    self.assertEqual(attributes_list, ['a', 'b'],
                                     f"Code snippet {i} did not return expected attributes list.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "_create_list_of_attributes_to_compare",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_create_list_of_attributes_to_compare",
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

        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_create_list_of_attributes_to_compare"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()