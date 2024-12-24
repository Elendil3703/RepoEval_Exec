import unittest
import json
import os
from typing import Any
import torch

TEST_RESULT_JSONL = "test_result.jsonl"

class TestInitFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[360]
        if not cls.code_snippet:
            raise ValueError("Expected a code snippet in the JSON file data")

    def test_init_function(self):
        """Dynamically test the __init__ function in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # 收集要写入 JSONL 的测试结果

        code = self.code_snippet

        # ------------------- 多一些逻辑检查 -------------------
        # 1) 静态检查：判断 snippet 中是否真的定义了 __init__
        if "def __init__" not in code:
            print(f"Code snippet: FAILED, '__init__' function not found in code.\n")
            failed_count += 1
            # 写入失败记录
            results.append({
                "function_name": "__init__",
                "code": code,
                "result": "failed"
            })
            self._write_results(results)
            return

        # ------------------- 动态执行并测试逻辑 -------------------
        exec_globals = {
            'torch': torch,
            'Any': Any,
        }
        exec_locals = {}

        try:
            # 动态执行代码片段
            exec(code, exec_globals, exec_locals)

            # 检查类和 __init__ 是否真的存在
            cls_name = [name for name, obj in exec_locals.items() if isinstance(obj, type)]
            if not cls_name:
                print(f"Code snippet: FAILED, no class found in exec_locals.\n")
                failed_count += 1
                results.append({
                    "function_name": "__init__",
                    "code": code,
                    "result": "failed"
                })
                self._write_results(results)
                return

            cls_instance = exec_locals[cls_name[0]](0.5, torch.device('cpu'))

            # Test default parameter
            self.assertEqual(cls_instance.dimension_to_project_features_to, 128,
                             f"Code snippet did not assign the default value to dimension_to_project_features_to.")
            print(f"Code snippet: PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "__init__",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "__init__",
                "code": code,
                "result": "failed"
            })

        self._write_results(results)

    def _write_results(self, results):
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

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()