import unittest
import json
import os
from typing import Any  # 确保注入的环境中有 Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGroundTruthInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[387]  # Get the 388th code snippet

    def test_ground_truth_init(self):
        """Test the __init__ method of the code snippet."""
        results = []  # Used to collect test results to write to JSONL

        for i, code in enumerate(self.code_snippet):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                # ------------------- 动态执行并测试逻辑 -------------------
                exec_globals = {
                    'Any': Any,  # Inject Any
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Ensure __init__ exists in the executed code
                    if '__init__' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, '__init__' not found in exec_locals.\n")
                        results.append({
                            "function_name": "__init__",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Instantiate the class with test parameters
                    test_instance = exec_locals['__init__'](5, nn_method=None)

                    # Assertions to check if attributes are correctly set
                    self.assertEqual(test_instance.n_nearest_neighbours, 5,
                                     f"Code snippet {i}: n_nearest_neighbours attribute not set correctly.")
                    self.assertIsNone(test_instance.nn_method,
                                      f"Code snippet {i}: nn_method attribute not set correctly to None.")

                    # Check the structure of lambda functions
                    self.assertTrue(callable(test_instance.imagelevel_nn),
                                    f"Code snippet {i}: imagelevel_nn is not callable.")
                    self.assertTrue(callable(test_instance.pixelwise_nn),
                                    f"Code snippet {i}: pixelwise_nn is not callable.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })

        # ============= Write the test results to test_result.jsonl =============
        # Read existing test_result.jsonl (Ignore if it doesn't exist)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "__init__"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__init__"
        ]

        # Append the new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()