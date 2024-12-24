import unittest
import json
import sys
import re
import os
from typing import Any, Dict, List, Union  # 确保注入的环境中导入必要类型

TEST_RESULT_JSONL = "test_result.jsonl"

class TestDataFrameInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[445]  # Get the 446th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the selected JSON array")

    def test_init_function(self):
        """Test the __init__ logic of DataFrame component."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # 收集要写入 JSONL 的测试结果

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # ------------------- 多一些逻辑检查 -------------------
                # Static check: Ensure __init__ is defined correctly
                init_pattern = r"def\s+__init__\s*\(self,\s*data:\s*dict\[Column,\s*Values\]\s*\|\s*None\s*=\s*None\)\s*->\s*None:"
                if not re.search(init_pattern, code):
                    print(f"Code snippet {i}: FAILED, incorrect signature for '__init__'.\n")
                    failed_count += 1
                    # 写入失败记录
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- 动态执行并测试逻辑 -------------------
                exec_globals = {
                    'sys': sys,
                    'Dict': Dict,
                    'List': List,
                    'Union': Union,
                    'Any': Any,
                    '_check_type': lambda x, y: True,  # Mock type checking
                    'PandasDataFrame': dict,  # 使用字典模拟 PandasDataFrame
                }
                exec_locals = {}

                try:
                    # 动态执行代码片段
                    exec(code, exec_globals, exec_locals)

                    # Check if class is instantiated correctly
                    class DataFrameTestPlaceholder:
                        pass

                    exec_locals['DataFrame'] = DataFrameTestPlaceholder

                    # Extend the class to test its initialization
                    class TestDataFrame(exec_locals['DataFrame']):
                        def __init__(self, data=None):
                            super().__init__(data)

                        @property
                        def data(self):
                            return self._data

                    # Test cases
                    df = TestDataFrame({"foo": [1, 2], "bar": ["A", "B"]})
                    self.assertIsInstance(df.data, dict, f"Code snippet {i} failed, data is not initialized as a dictionary.")
                    self.assertEqual(df.data["foo"], [1, 2], f"Code snippet {i} failed, 'foo' column data incorrect.")
                    self.assertEqual(df.data["bar"], ["A", "B"], f"Code snippet {i} failed, 'bar' column data incorrect.")

                    # Test with None
                    df_empty = TestDataFrame(None)
                    self.assertEqual(df_empty.data, {}, f"Code snippet {i} failed, initialized data should be empty dict.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
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

        # 删除 function_name == "__init__" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__init__"
        ]

        # 将新结果附加
        existing_records.extend(results)

        # 重写 test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()