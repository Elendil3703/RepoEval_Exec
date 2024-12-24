import unittest
import json
import os
import numpy as np
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"


class TestCastUint8ImagesToFloat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        # Get the 54th code snippet (index 53 in zero-indexed terms)
        cls.code_snippet = data[53]  
        if not cls.code_snippet:
            raise ValueError("Expected a valid code snippet at index 53 in the JSON array")

    def test_cast_uint8_images_to_float(self):
        """Test the cast_uint8_images_to_float function logic in the code snippet."""
        exec_globals = {
            'np': np,
            'Any': Any
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(self.code_snippet, exec_globals, exec_locals)

            # Check that the function is correctly defined
            if 'cast_uint8_images_to_float' not in exec_locals:
                self.fail("Function 'cast_uint8_images_to_float' is not defined in the code snippet.")

            cast_fn = exec_locals['cast_uint8_images_to_float']

            # Test cases
            test_cases = [
                (np.array([[0, 127, 255]], dtype=np.uint8), np.array([[0., 127/255, 1.]], dtype=np.float32)),
                (np.array([[1, 2, 3]], dtype=np.float32), np.array([[1, 2, 3]], dtype=np.float32)),
                (np.array([[0, 255]], dtype=np.int32), np.array([[0, 255]], dtype=np.int32))
            ]

            passed_count = 0
            failed_count = 0
            results = []

            for idx, (input_image, expected_output) in enumerate(test_cases):
                with self.subTest(test_index=idx):
                    print(f"Running test for test case {idx}...")
                    result = cast_fn(input_image)
                    
                    if np.array_equal(result, expected_output):
                        print(f"Test case {idx}: PASSED\n")
                        passed_count += 1
                        results.append({
                            "function_name": "cast_uint8_images_to_float",
                            "test_case_index": idx,
                            "result": "passed"
                        })
                    else:
                        print(f"Test case {idx}: FAILED - Expected {expected_output}, got {result}\n")
                        failed_count += 1
                        results.append({
                            "function_name": "cast_uint8_images_to_float",
                            "test_case_index": idx,
                            "result": "failed"
                        })

            # Final Summary
            print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(test_cases)}\n")

            self.assertEqual(passed_count + failed_count, len(test_cases), "Test count mismatch!")

            # Writing results to test_result.jsonl
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
                if rec.get("function_name") != "cast_uint8_images_to_float"
            ]

            existing_records.extend(results)

            with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
                for record in existing_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print("Results have been written to test_result.jsonl")

        except Exception as e:
            self.fail(f"Failed with error: {e}")

if __name__ == "__main__":
    unittest.main()