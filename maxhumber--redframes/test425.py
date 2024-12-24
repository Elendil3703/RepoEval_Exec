import unittest
import json
import os
from pandas import DataFrame
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestDedupeFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[424]  # Get the specific JSON element
        if len(cls.code_snippet) < 1:
            raise ValueError("Expected code snippet at index 424 in JSON array")

    def test_dedupe_function(self):
        """Dynamically test dedupe function from provided code."""
        
        code = self.code_snippet
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # 收集要写入 JSONL 的测试结果

        exec_globals = {
            'Any': Any,
            'PandasDataFrame': DataFrame,
        }
        exec_locals = {}

        try:
            # 动态执行代码片段
            exec(code, exec_globals, exec_locals)

            # 检查 dedupe 是否存在
            if 'dedupe' not in exec_locals:
                print("Code snippet: FAILED, 'dedupe' not found in exec_locals.")
                failed_count += 1
                results.append({
                    "function_name": "dedupe",
                    "code": code,
                    "result": "failed"
                })
                return

            dedupe = exec_locals['dedupe']

            # 准备数据以供测试
            data = {'A': [1, 1, 2, 3, 3], 'B': [5, 5, 6, 7, 7]}
            df = DataFrame(data)

            # 测试1: 默认参数
            result_df = dedupe(df)
            expected_df = DataFrame({'A': [1, 2, 3], 'B': [5, 6, 7]})
            self.assertTrue(result_df.equals(expected_df), "Test with default parameters failed.")

            # 测试2: 使用特定的列去除重复项
            result_df = dedupe(df, columns='A')
            expected_df = DataFrame({'A': [1, 2, 3], 'B': [5, 6, 7]})
            self.assertTrue(result_df.equals(expected_df), "Test with 'A' column failed.")

            # 测试3: 空DataFrame无动作
            empty_df = DataFrame({'A': [], 'B': []})
            result_df = dedupe(empty_df)
            expected_df = empty_df
            self.assertTrue(result_df.equals(expected_df), "Test with empty DataFrame failed.")

            print("Code snippet: PASSED all assertions.")
            passed_count += 1
            results.append({
                "function_name": "dedupe",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}")
            failed_count += 1
            results.append({
                "function_name": "dedupe",
                "code": code,
                "result": "failed"
            })

        # 最终统计信息
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed.")

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

        # 删除 function_name == "dedupe" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "dedupe"
        ]

        # 将新结果附加
        existing_records.extend(results)

        # 重写 test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()