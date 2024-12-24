import unittest
import json
import sys
import re
import os
from unittest.mock import Mock
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestWrapUnetFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[102]  # Get the desired JSON element (103rd snippet)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the provided JSON data")

    def test_wrap_unet(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # 收集要写入 JSONL 的测试结果

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # ------------------- 多一些逻辑检查 -------------------
                # 1) 静态检查：判断 snippet 中是否真的定义了 wrap_unet
                if "def wrap_unet" not in code:
                    print(f"Code snippet {i}: FAILED, function 'wrap_unet' not defined.\n")
                    failed_count += 1
                    # 写入失败记录
                    results.append({
                        "function_name": "wrap_unet",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                func_pattern = r"def\s+wrap_unet\s*\("
                if not re.search(func_pattern, code):
                    print(f"Code snippet {i}: FAILED, incorrect signature for 'wrap_unet'.\n")
                    failed_count += 1
                    # 写入失败记录
                    results.append({
                        "function_name": "wrap_unet",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- 动态执行并测试逻辑 -------------------
                exec_globals = {
                    'sys': sys,
                    'Mock': Mock,
                    'Any': Any,  # 注入 Any
                }
                exec_locals = {}

                try:
                    # 动态执行代码片段
                    exec(code, exec_globals, exec_locals)

                    # 检查 wrap_unet 是否真的存在
                    if 'wrap_unet' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'wrap_unet' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "wrap_unet",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # 设置 mock 对象和属性
                    mock_self = Mock()
                    mock_optimizer = Mock()
                    mock_scheduler = Mock()
                    mock_unet = Mock()

                    mock_self.imagen.get_unet.return_value = mock_unet
                    setattr(mock_self, 'optim0', mock_optimizer)
                    setattr(mock_self, 'scheduler0', mock_scheduler)

                    # 执行 wrap_unet 函数
                    exec_locals['wrap_unet'](mock_self, 1)

                    # 检查是否设置了标识符
                    self.assertTrue(getattr(mock_self, 'one_unet_wrapped', False), f"Code snippet {i} did not set 'one_unet_wrapped' to True.")
                    
                    # 检查 accelerator.prepare 是否被调用
                    mock_self.accelerator.prepare.assert_called()

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "wrap_unet",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "wrap_unet",
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

        # 删除 function_name == "wrap_unet" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "wrap_unet"
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