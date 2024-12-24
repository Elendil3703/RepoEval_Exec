import unittest
import json
import os
import pandas as pd
from typing import Any  # 确保注入的环境中有 Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestSpreadFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[432]  # Get the 433rd JSON element (index 432)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 433rd JSON array")

    def test_spread_function(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # 收集要写入 JSONL 的测试结果
        
        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                # ------------------- 静态检查 -------------------
                if "def spread" not in code:
                    print(f"Code snippet {i}: FAILED, function 'spread' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "spread",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                func_pattern = r"def\s+spread\s*\("
                if not re.search(func_pattern, code):
                    print(f"Code snippet {i}: FAILED, incorrect signature for 'spread'.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "spread",
                        "code": code,
                        "result": "failed"
                    })
                    continue
                
                # ------------------- 动态执行并测试逻辑 -------------------
                exec_globals = {
                    'pd': pd,
                    'uuid': __import__('uuid'),
                    'Any': Any,  # 注入 Any
                }
                exec_locals = {}

                try:
                    # 动态执行代码片段
                    exec(code, exec_globals, exec_locals)

                    # 检查 spread 是否真的已定义
                    if 'spread' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'spread' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "spread",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # 准备测试数据
                    data = {
                        'A': ['foo', 'foo', 'bar', 'bar'],
                        'B': ['one', 'two', 'one', 'two'],
                        'C': [1, 2, 3, 4]
                    }
                    df = pd.DataFrame(data)

                    # 调用 spread 函数
                    result_df = exec_locals['spread'](df, 'B', 'C')

                    # 预期结果
                    expected_data = {
                        'A': ['bar', 'foo'],
                        'one': [3, 1],
                        'two': [4, 2]
                    }
                    expected_df = pd.DataFrame(expected_data)

                    # 验证结果
                    pd.testing.assert_frame_equal(result_df, expected_df)
                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "spread",
                        "code": code,
                        "result": "passed"
                    })

                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "spread",
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

        # 删除 function_name == "spread" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "spread"
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