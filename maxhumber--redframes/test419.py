import unittest
import json
import os
from typing import Union

TEST_RESULT_JSONL = "test_result.jsonl"

class PandasDataFrame:
    # Mock class for PandasDataFrame
    def __init__(self, data, columns):
        self.data = data
        self.columns = columns

    def agg(self, **kwargs):
        column = list(kwargs.keys())[0]
        func = kwargs[column][1]
        new_data = [{column: func(row[column])} for row in self.data]
        return PandasDataFrame(new_data, [column])

    def __getitem__(self, columns):
        return PandasDataFrame([{col: row[col] for col in columns} for row in self.data], columns)

    def reset_index(self, drop):
        return self

class PandasGroupedFrame:
    # Mock class for PandasGroupedFrame
    def __init__(self, obj):
        self.obj = obj

def _check_type(value, expected_type):
    if not isinstance(value, expected_type):
        raise TypeError(f"Expected {expected_type}, got {type(value)}")

def pack(
    df: Union[PandasDataFrame, PandasGroupedFrame], column: str, sep: str
) -> PandasDataFrame:
    _check_type(column, str)
    _check_type(sep, str)
    order = df.obj.columns if isinstance(df, PandasGroupedFrame) else df.columns  # type: ignore
    df = df.agg(**{column: (column, lambda x: str(x) + sep)})  # Simplified agg mock
    df = df[[col for col in df.columns if col in order]]
    df = df.reset_index(drop=True)
    return df

class TestPackFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[418]  # Get the 419th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 419th JSON array")

    def test_pack_function(self):
        """Test the pack function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collecting test results

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                exec_globals = {
                    'PandasDataFrame': PandasDataFrame,
                    'PandasGroupedFrame': PandasGroupedFrame,
                    '_check_type': _check_type,
                    'pack': None,  # To be injected
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)
                    pack_function = exec_globals.get('pack')

                    if not callable(pack_function):
                        print(f"Code snippet {i}: FAILED, 'pack' function not found or callable.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "pack",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test cases
                    df_sample = PandasDataFrame(
                        data=[{"col1": "data1", "col2": "data2"}, {"col1": "data3", "col2": "data4"}],
                        columns=["col1", "col2"]
                    )

                    result_df = pack_function(df_sample, "col1", "-")
                    assert result_df.data[0]["col1"] == "data1-"

                    print(f"Code snippet {i}: PASSED.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "pack",
                        "code": code,
                        "result": "passed"
                    })

                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "pack",
                        "code": code,
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write test results to test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Delete old records for the "pack" function
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "pack"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()