import unittest
import json
import os
from typing import Dict  # 确保注入的环境中有 Dict

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGroundTruthFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[392]  # Get the 393rd JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 393rd JSON array")

    def test_call_function(self):
        """Test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # 收集要写入 JSONL 的测试结果

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                exec_globals = {
                    'Dict': Dict,
                }
                exec_locals = {}

                try:
                    # 动态执行代码片段
                    exec(code, exec_globals, exec_locals)

                    # 检查 __call__ 是否真的存在
                    if '__call__' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, '__call__' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "__call__",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # 模拟测试用例
                    # 使用一个假设的 key_pattern 和 state_dict 来测试 __call__
                    key_pattern = "weight.*"
                    state_dict = {
                        'weight1': 1,
                        'weight2': 2,
                        'bias': 0,
                    }

                    # 应该在执行 exec 时定义
                    _unix_pattern_to_parameter_names = exec_locals.get('_unix_pattern_to_parameter_names')
                    if not _unix_pattern_to_parameter_names:
                        raise ValueError("Function '_unix_pattern_to_parameter_names' not found.")

                    class TestClass:
                        def __init__(self, key_pattern):
                            self.key_pattern = key_pattern

                        __call__ = staticmethod(exec_locals['__call__'])

                    # 创建测试对象
                    test_obj = TestClass(key_pattern)

                    # 调用 __call__ 函数
                    result = test_obj.__call__(state_dict)

                    # 预期结果
                    expected_result = {
                        'weight1': 1,
                        'weight2': 2
                    }

                    self.assertEqual(
                        result,
                        expected_result,
                        f"Code snippet {i}: Failed to return the correct filtered state_dict.",
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "__call__",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__call__",
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

        # 删除 function_name == "__call__" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__call__"
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