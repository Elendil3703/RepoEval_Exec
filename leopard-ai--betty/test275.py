import unittest
import json
import sys
import re
import os
from typing import Dict, Any  # 用于类型提示，确保任何注入的环境都有

TEST_RESULT_JSONL = "test_result.jsonl"

class TestLogFromLossDictResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 加载 JSON 文件
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[274]  # 获取第275组代码变为索引274的json元素
        if not cls.code_snippet:
            raise ValueError("Expected a code snippet in the 275th JSON array")

    def test_log_from_loss_dict(self):
        """Dynamically test log_from_loss_dict with various test cases."""
        passed_count = 0  # 通过的测试计数
        failed_count = 0  # 失败的测试计数
        results = []      # 收集要写入 JSONL 的测试结果

        code = self.code_snippet

        print("Running dynamic test for log_from_loss_dict...")

        # 检查函数签名
        func_pattern = r"def\s+log_from_loss_dict\s*\("
        if not re.search(func_pattern, code):
            print("FAILED: Function signature for 'log_from_loss_dict' not found.\n")
            failed_count += 1

            results.append({
                "function_name": "log_from_loss_dict",
                "code": code,
                "result": "failed"
            })
            self.assertTrue(False, "Function 'log_from_loss_dict' not found in code.")
            return

        # 默认添加空检测下的torch环境
        assert "torch" in code, "Code does not have a check for torch module!"

        exec_globals = {
            'torch': __import__('torch'),  # 假设 torch 在导入时可用
        }
        exec_locals = {}

        try:
            # 动态执行代码片段
            exec(code, exec_globals, exec_locals)

            # 检查 log_from_loss_dict 是否存在
            if 'log_from_loss_dict' not in exec_locals:
                print("FAILED: 'log_from_loss_dict' not found in exec_locals.\n")
                failed_count += 1
                results.append({
                    "function_name": "log_from_loss_dict",
                    "code": code,
                    "result": "failed"
                })
                self.assertTrue(False, "'log_from_loss_dict' not found after execution.")
                return

            log_from_loss_dict = exec_locals['log_from_loss_dict']

            # ---- 开始一系列针对 log_from_loss_dict 的测试案例 ----

            # 测试样例1: 简单字典
            loss_dict = {'loss': 0.123}
            expected_output = "loss: 0.123"
            self.assertEqual(log_from_loss_dict(loss_dict), expected_output)

            # 测试样例2: 多个标量值
            loss_dict = {'loss': 0.123, 'accuracy': 0.95}
            expected_output = "loss: 0.123 || accuracy: 0.95"
            self.assertEqual(log_from_loss_dict(loss_dict), expected_output)

            # 测试样例3: 复杂嵌套带张量的字典
            loss_dict = {
                'loss': [__import__('torch').tensor(0.123), __import__('torch').tensor(0.456)],
                'metrics': {'accuracy': __import__('torch').tensor(0.789)},
            }
            expected_output = "loss_0: 0.123 || loss_1: 0.456 || metrics_0: 0.789"
            self.assertEqual(log_from_loss_dict(loss_dict), expected_output)

            print("PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "log_from_loss_dict",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "log_from_loss_dict",
                "code": code,
                "result": "failed"
            })

        # 最终统计信息
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed in total.\n")

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

        # 删除函数名为 log_from_loss_dict 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "log_from_loss_dict"
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