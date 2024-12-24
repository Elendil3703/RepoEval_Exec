import unittest
import json
import sys
import re
import os
from typing import Any  # 确保注入的环境中有 Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestMockModelFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[304]  # Get the 305th JSON element (index 304)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the correct JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # 收集要写入 JSONL 的测试结果

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # ------------------- 多一些逻辑检查 -------------------
                # 1) 静态检查：判断 snippet 中是否定义了 mock_model_function
                if "mock_model_function" not in code:
                    print(f"Code snippet {i}: FAILED, 'mock_model_function' not found in code.\n")
                    failed_count += 1
                    # 写入失败记录
                    results.append({
                        "function_name": "mock_model_function",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                func_pattern = r"def\s+mock_model_function\s*\("
                if not re.search(func_pattern, code):
                    print(f"Code snippet {i}: FAILED, incorrect signature for 'mock_model_function'.\n")
                    failed_count += 1
                    # 写入失败记录
                    results.append({
                        "function_name": "mock_model_function",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- 动态执行并测试逻辑 -------------------
                exec_globals = {
                    'numpyro': __import__('numpyro'),
                    'seasonality': type('MockSeasonality', (object,), {'intra_week_seasonality': lambda self, data, custom_priors: data}),
                }
                exec_locals = {}

                try:
                    # 动态执行代码片段
                    exec(code, exec_globals, exec_locals)

                    # 检查 mock_model_function 是否真的存在
                    if 'mock_model_function' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'mock_model_function' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "mock_model_function",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # 使用 mock_model_function 测试其行为
                    test_data = {"sample": "data"}
                    result = exec_locals['mock_model_function'](test_data)

                    # 检查结果与期望数据是否匹配
                    self.assertEqual(
                        result,
                        test_data,
                        f"Code snippet {i} did not return the expected output.",
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "mock_model_function",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "mock_model_function",
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

        # 删除 function_name == "mock_model_function" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "mock_model_function"
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