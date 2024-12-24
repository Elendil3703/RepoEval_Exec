import unittest
import json
import os
import pandas as pd
from typing import Any  # 确保注入的环境中有 Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestMeltFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        cls.code_snippet = data[433]  # Get the 434th JSON element (index 433)

    def test_melt_function(self):
        """Test the _melt function from the provided code snippet."""
        # Assume code snippet is a single string containing the function and required imports
        code = self.code_snippet

        # Inject environment
        exec_globals = {
            'pd': pd,
            'PandasDataFrame': pd.DataFrame,
            'Any': Any,  # 注入 Any
        }
        exec_locals = {}

        try:
            # Execute the provided code containing the _melt function
            exec(code, exec_globals, exec_locals)

            # Retrieve the _melt function from the execution locals
            if '_melt' not in exec_locals:
                self.fail("The `_melt` function was not found in the executed code.")

            _melt = exec_locals['_melt']

            # Define a DataFrame and expected result for testing
            df_input = pd.DataFrame({
                'A': ['foo', 'bar'],
                'B': ['one', 'two'],
                'C': [1, 2],
                'D': [3, 4]
            })
            df_expected = pd.DataFrame({
                'A': ['foo', 'bar', 'foo', 'bar'],
                'B': ['one', 'two', 'one', 'two'],
                'variable': ['C', 'C', 'D', 'D'],
                'value': [1, 2, 3, 4]
            })

            # Call the _melt function with test parameters
            df_result = _melt(
                df_input,
                cols_to_keep=['A', 'B'],
                cols_to_gather=['C', 'D'],
                into=('variable', 'value')
            )

            # Assert the result matches expected DataFrame
            pd.testing.assert_frame_equal(df_result, df_expected)

            # If success, write passed result
            result = {
                "function_name": "_melt",
                "code": code,
                "result": "passed"
            }
        except Exception as e:
            # If any error occurs, write failed result with error message
            result = {
                "function_name": "_melt",
                "code": code,
                "result": "failed",
                "error": str(e)
            }

        # ============== Append result to test_result.jsonl ==============
        # Read existing records, if any
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        existing_records.append(json.loads(line))

        # Remove old records with function_name == "_melt"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_melt"
        ]

        # Add new result
        existing_records.append(result)

        # Write updated results back to the JSONL file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()