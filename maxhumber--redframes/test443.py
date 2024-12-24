import unittest
import json
import os
from pandas import DataFrame as PandasDataFrame
from typing import Any  # 确保注入的环境中有 Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestWrapFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[442]  # Get the 443rd JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 443rd JSON array")

    def test_wrap_function(self):
        """Dynamically test all code snippets for the wrap function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # 收集要写入 JSONL 的测试结果

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # ------------------- 静态检查 -------------------
                if "def wrap" not in code:
                    print(f"Code snippet {i}: FAILED, function 'wrap' not found.\n")
                    failed_count += 1
                    # 写入失败记录
                    results.append({
                        "function_name": "wrap",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- 动态执行并测试逻辑 -------------------
                exec_globals = {
                    'PandasDataFrame': PandasDataFrame,
                    'DataFrame': DataFrame,  # Placeholder for the custom DataFrame
                    'Any': Any,  # 注入 Any
                    '_check_type': lambda df, _type: isinstance(df, _type),
                    '_check_index': lambda df: True,
                    '_check_columns': lambda df: True,
                }
                exec_locals = {}

                try:
                    # 动态执行代码片段
                    exec(code, exec_globals, exec_locals)

                    # 确保 wrap 函数已定义
                    if 'wrap' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'wrap' function not executed correctly.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "wrap",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # 模拟输入并验证返回的 DataFrame
                    class DataFrame:
                        def __init__(self):
                            self._data = None

                    pdf = PandasDataFrame({"foo": range(10)})
                    rdf = exec_locals['wrap'](pdf)

                    # Assertions for expected behavior
                    self.assertIsInstance(
                        rdf, DataFrame,
                        f"Code snippet {i}: wrap function did not return a DataFrame instance."
                    )
                    self.assertTrue(
                        rdf._data.equals(pdf),
                        f"Code snippet {i}: wrap function did not correctly copy data from PandasDataFrame."
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "wrap",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "wrap",
                        "code": code,
                        "result": "failed"
                    })

        # 最终统计信息
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

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

        # 删除 function_name == "wrap" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "wrap"
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