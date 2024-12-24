import unittest
import json
import sys
import re
import os
from typing import Any  # 确保注入的环境中有 Any

TEST_RESULT_JSONL = "test_result.jsonl"

_METHODS = {}

# 为了避免抛出 'MethodConfig is not defined' 错误，我们简单地创建一个占位类
class MethodConfig:
    pass

def get_method(name: str) -> MethodConfig:
    """Return constructor for specified method config"""
    name = name.lower()
    if name in _METHODS:
        return _METHODS[name]
    else:
        raise Exception("Error: Trying to access a method that has not been registered")

class TestCarperAITrlxResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 加载 JSON 文件
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[6]  # 获取第七个 JSON 元素（索引 6）
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 7th JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # 通过的测试计数
        failed_count = 0  # 失败的测试计数
        results = []      # 收集要写入 JSONL 的测试结果

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # ------------------- 多一些逻辑检查 -------------------
                # 1) 静态检查：判断代码片段中是否定义了 _METHODS 和 get_method
                if "_METHODS" not in code:
                    print(f"Code snippet {i}: FAILED, '_METHODS' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "get_method",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                if "def get_method" not in code:
                    print(f"Code snippet {i}: FAILED, function 'get_method' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "get_method",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                func_pattern = r"def\s+get_method\s*\("
                if not re.search(func_pattern, code):
                    print(f"Code snippet {i}: FAILED, incorrect signature for 'get_method'.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "get_method",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- 动态执行并测试逻辑 -------------------
                exec_globals = {
                    'sys': sys,
                    '_METHODS': _METHODS,  # 确保 _METHODS 在执行环境中
                    'Any': Any,  # 注入 Any
                    'MethodConfig': MethodConfig  # 注入 MethodConfig 占位符类
                }
                exec_locals = {}

                try:
                    # 动态执行代码片段
                    exec(code, exec_globals, exec_locals)

                    # 检查 get_method 是否存在于 exec_locals 中
                    if 'get_method' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'get_method' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "get_method",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # 测试：注册一个方法并验证 get_method 是否能正确返回
                    class TestMethod(MethodConfig):
                        pass

                    _METHODS['test_method'] = TestMethod

                    # 测试 get_method 能否成功返回注册的 MethodConfig
                    retrieved_method = exec_locals['get_method']("test_method")
                    self.assertEqual(
                        retrieved_method, TestMethod,
                        f"Code snippet {i}: 'get_method' did not return the expected class for 'test_method'."
                    )

                    # 测试：访问未注册的方法是否抛出异常
                    with self.assertRaises(Exception):
                        exec_locals['get_method']("non_existent_method")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "get_method",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "get_method",
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

        # 删除 function_name == "get_method" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "get_method"
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