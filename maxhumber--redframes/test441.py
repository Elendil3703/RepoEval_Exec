import unittest
import json
import os
from pandas import DataFrame
import pandas as pd

TEST_RESULT_JSONL = "test_result.jsonl"

class TestSaveFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[440]  # Get the 441st JSON element (index 440)
        if len(cls.code_snippet) < 1:
            raise ValueError("Expected at least one code snippet")

    def run_test(self, code):
        """Run a specific test for the 'save' function with the given code snippet."""
        results = []  # To collect results
        
        # Prepare to execute the code
        exec_globals = {
            'DataFrame': DataFrame,
            '_check_type': lambda x, t: isinstance(x, t),
            '_check_file': lambda p: None,  # Placeholder
            'pd': pd,
        }
        exec_locals = {}

        try:
            # Dynamically execute code snippet
            exec(code, exec_globals, exec_locals)

            # Check if 'save' function is defined
            if 'save' not in exec_locals:
                print("FAILED: 'save' function not defined in code.\n")
                results.append({
                    "function_name": "save",
                    "code": code,
                    "result": "failed"
                })
                return results

            # Test 'save' function
            save_func = exec_locals['save']

            # Create a DataFrame and a file path for testing
            df_to_save = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            test_path = "test_output.csv"

            # Call the save function
            save_func(df_to_save, test_path)

            # Verify if the file was created
            self.assertTrue(os.path.exists(test_path), "Failed to create output CSV file.")

            # Read the file back to verify the content
            df_loaded = pd.read_csv(test_path)
            pd.testing.assert_frame_equal(df_to_save, df_loaded, "Mismatch between saved and loaded data.")

            # Cleanup
            if os.path.exists(test_path):
                os.remove(test_path)

            print("PASSED all assertions.\n")
            results.append({
                "function_name": "save",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"Code failed with error: {e}\n")
            results.append({
                "function_name": "save",
                "code": code,
                "result": "failed"
            })

        return results

    def test_save_function(self):
        results = self.run_test(self.code_snippet)

        # ================= 将测试结果写入 test_result.jsonl =================
        # Read existing test_result.jsonl (if present)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Delete old records with function_name == "save"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "save"
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