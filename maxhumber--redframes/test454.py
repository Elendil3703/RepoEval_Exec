import unittest
import json
import os
from typing import Any  # 确保注入的环境中有 Any
from pandas import DataFrame as PandasDataFrame, RangeIndex as PandasRangeIndex

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCheckIndexFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file and get the specific code snippet
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[453]  # Get the 454th JSON element (index 453)
    
    def test_check_index(self):
        """Dynamically test the _check_index function in the JSON."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # 收集要写入 JSONL 的测试结果

        for i, code in enumerate([self.code_snippet]):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                # Prepare the exec environment
                exec_globals = {
                    'PandasDataFrame': PandasDataFrame,
                    'PandasRangeIndex': PandasRangeIndex
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check that _check_index is defined
                    if '_check_index' not in exec_locals:
                        raise ValueError("_check_index not found in executed code.")

                    # Define some dataframes for testing
                    valid_df = PandasDataFrame(index=PandasRangeIndex(start=0, stop=10, step=1, name=None))
                    invalid_name_df = PandasDataFrame(index=PandasRangeIndex(start=0, stop=10, step=1, name='name'))
                    invalid_type_df = PandasDataFrame(index=[0, 1, 2, 3, 4])
                    invalid_start_df = PandasDataFrame(index=PandasRangeIndex(start=1, stop=11, step=1))
                    invalid_step_df = PandasDataFrame(index=PandasRangeIndex(start=0, stop=10, step=2))

                    # Execute tests
                    try:
                        exec_locals['_check_index'](valid_df)
                        test_passed = True
                    except IndexError:
                        test_passed = False

                    self.assertTrue(test_passed, f"Code snippet {i}: Valid DataFrame should not raise an error.")

                    test_cases = [
                        (invalid_name_df, "IndexError: must be unnamed"),
                        (invalid_type_df, "IndexError: must be range"),
                        (invalid_start_df, "IndexError: must start at 0"),
                        (invalid_step_df, "IndexError: must step by 1"),
                    ]

                    for df, expected_error in test_cases:
                        with self.assertRaises(IndexError, msg=f"Expected {expected_error} for DataFrame: {df}"):
                            exec_locals['_check_index'](df)

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "_check_index",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_check_index",
                        "code": code,
                        "result": "failed"
                    })

        # 最终统计信息
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 1\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # ============= 将测试结果写入 test_result.jsonl =============
        # 读取现有 test_result.jsonl（若不存在则忽略）
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # 删除 function_name == "_check_index" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_check_index"
        ]

        # 将新结果附加
        existing_records.extend(results)

        # 重写 test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()