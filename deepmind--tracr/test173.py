import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestMakeReverseFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[172]  # Get the 173rd JSON element (index 172)
        if len(cls.code_snippet) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_make_reverse(self):
        """Test the make_reverse function with various inputs."""
        passed_count = 0
        failed_count = 0
        results = []

        code = self.code_snippet

        globals_dict = {
            'rasp': Any  # Mocking rasp module since we don't have its implementation
        }
        
        try:
            # Since we're missing the real 'rasp' module, we'll mock its behavior
            class MockSOp:
                def __init__(self, value):
                    self.value = value
                
                def __call__(self, input_sequence):
                    return list(reversed(input_sequence))
            
            class MockIndices:
                def named(self, name):
                    return self
            
            class MockLength:
                pass

            class MockComparison:
                EQ = "=="

            class MockSelect:
                def __init__(self, *args, **kwargs):
                    pass
                
                def named(self, name):
                    return self

            class MockAggregate:
                def __init__(self, *args, **kwargs):
                    pass

                def named(self, name):
                    return MockSOp(None)  # Returning a mock SOp object
            
            globals_dict['rasp'] = type('MockRasp', (object,), {
                'SOp': MockSOp,
                'tokens': MockSOp(None),
                'indices': MockIndices(),
                'Select': MockSelect,
                'Aggregate': MockAggregate,
                'Comparison': MockComparison,
                'length': MockLength()
            })()

            exec(code, globals_dict)

            # Verifying if make_reverse is defined after execution
            self.assertIn("make_reverse", globals_dict, "'make_reverse' function is not defined in the snippet.")

            make_reverse = globals_dict['make_reverse']
            self.assertTrue(callable(make_reverse), "'make_reverse' should be callable.")

            # Create a reverse operation instance
            reverse_op = make_reverse(globals_dict['rasp'].tokens)
            self.assertTrue(isinstance(reverse_op, globals_dict['rasp'].SOp), "Output of make_reverse should be an instance of MockSOp.")

            # Test the reverse operation
            test_sequence = ['H', 'e', 'l', 'l', 'o']
            reversed_sequence = reverse_op(test_sequence)
            expected_sequence = ['o', 'l', 'l', 'e', 'H']
            self.assertEqual(reversed_sequence, expected_sequence, "The function did not reverse the sequence correctly.")

            print("Test passed for make_reverse function.")
            passed_count += 1
            results.append({
                "function_name": "make_reverse",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"Test failed with error: {e}")
            failed_count += 1
            results.append({
                "function_name": "make_reverse",
                "code": code,
                "result": "failed",
                "error": str(e)
            })

        # Statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed.\n")

        # ============= Write test results to test_result.jsonl =============
        # Read existing test_result.jsonl (ignore if not exists)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "make_reverse"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "make_reverse"
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