import unittest
import json
import os
from typing import List, Any  # Making sure List and Any are available for testing
import rasp  # Assuming the module is available in the testing environment

TEST_RESULT_JSONL = "test_result.jsonl"

class TestDetectPatternFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[180]  # Get the 181st JSON element (index 180)
        if not cls.code_snippet:
            raise ValueError("The code snippet is empty or could not be found.")

    def test_detect_pattern(self):
        """Test the provided detect_pattern function with specific scenarios."""

        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        # Prepare the execution environment
        exec_globals = {
            'rasp': rasp,
            'shift_by': rasp.shift_by,  # Assuming shift_by is defined or imported
        }
        exec_locals = {}

        try:
            # Execute the code snippet to define detect_pattern
            exec(self.code_snippet, exec_globals, exec_locals)
            detect_pattern = exec_locals.get('detect_pattern')

            # Check if detect_pattern is correctly defined
            self.assertIsNotNone(detect_pattern, "The function detect_pattern is not defined.")

            # Specific tests for detect_pattern
            mock_sop = rasp.SOp(["a", "b", "a", "b", "c"])  # Example sequence

            # Test 1: Normal pattern match
            result = detect_pattern(mock_sop, "abc")(mock_sop)
            self.assertEqual(result, [None, None, None, None, True], 
                             "Pattern 'abc' should be detected at position 4.")  # Adjust based on expected behavior
            results.append({"function_name": "detect_pattern", "input": ["abcabc", "abc"], "result": "passed"})

            # Test 2: Pattern not present
            result = detect_pattern(mock_sop, "ac")(mock_sop)
            self.assertEqual(result, [None, None, False, False, False],
                             "Pattern 'ac' should not be detected.")  # Adjust based on expected behavior
            results.append({"function_name": "detect_pattern", "input": ["abcabc", "ac"], "result": "passed"})

            # Test 3: Error on empty pattern
            with self.assertRaises(ValueError, msg="Should raise ValueError on empty pattern."):
                detect_pattern(mock_sop, "")(mock_sop)
            results.append({"function_name": "detect_pattern", "input": ["abcabc", ""], "result": "passed"})

            passed_count += 3  # Adjust based on the number of tests
        except Exception as e:
            print(f"Test failed with error: {e}")
            failed_count += 1
            results.append({"function_name": "detect_pattern", "input": None, "result": "failed"})

        # Results summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {passed_count + failed_count}\n")
        self.assertEqual(passed_count + failed_count, 3, "Test count mismatch!")  # Adjust based on the number of tests

        # ============= Write test results to test_result.jsonl =============
        # Load existing records (if any)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for "detect_pattern"
        existing_records = [rec for rec in existing_records if rec.get("function_name") != "detect_pattern"]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()