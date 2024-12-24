import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestPostInitFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[222]  # Get the 223rd JSON element (index 222)
        if len(cls.code_snippet) < 1:
            raise ValueError("Expected a valid code snippet in the JSON array")

    def test_post_init(self):
        """Test the __post_init__ function with additional logical checks."""
        passed_count = 0
        failed_count = 0
        results = []

        code = self.code_snippet

        # ------------------- Check Import Dependencies -------------------
        if "import bases" not in code:
            print("Code snippet: FAILED, 'import bases' not found in code.\n")
            failed_count += 1
            results.append({
                "function_name": "__post_init__",
                "code": code,
                "result": "failed"
            })
            return

        # ------------------- Dynamic Execution and Testing -------------------
        exec_globals = {
            'bases': __import__("bases"),  # Assuming 'bases' module needs to be imported
            'Any': Any,
        }
        exec_locals = {}

        try:
            # Execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Instantiate an object and call __post_init__
            test_obj = exec_locals['TestClass']()  # Assuming a class name e.g., TestClass
            test_obj.__post_init__()

            # Check that residual_space is correctly set
            self.assertIsNotNone(test_obj.residual_space, "residual_space is not set.")

            # Check that spaces are subspaces of the residual_space
            assert_methods = [
                (test_obj.w_qk.left_space.issubspace, "w_qk.left_space"),
                (test_obj.w_qk.right_space.issubspace, "w_qk.right_space"),
                (test_obj.w_ov.input_space.issubspace, "w_ov.input_space"),
                (test_obj.w_ov.output_space.issubspace, "w_ov.output_space"),
            ]

            for assert_method, space_name in assert_methods:
                self.assertTrue(
                    assert_method(test_obj.residual_space),
                    f"{space_name} is not a subspace of the residual_space."
                )

            print("Code snippet: PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "__post_init__",
                "code": code,
                "result": "passed"
            })

        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "__post_init__",
                "code": code,
                "result": "failed"
            })

        # Final Assertion
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")  # Only one test case

        # Write Results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__post_init__"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()