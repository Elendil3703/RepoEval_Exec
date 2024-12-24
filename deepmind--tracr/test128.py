import unittest
import json
import sys
import os
from typing import Callable, Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestInitFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[127]  # Get the 128th JSON element (index 127)
        if len(cls.code_snippet) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_init_function(self):
        """Dynamically test the __init__ function in the extracted code snippet."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []

        code = self.code_snippet

        # ------------------- 多一些逻辑检查 -------------------
        print("Running test for __init__ function...")

        # 1) 静态检查：判断 snippet 中是否定义了 __init__
        if "def __init__" not in code:
            print("Code snippet: FAILED, '__init__' function not found in code.\n")
            failed_count += 1
            # 写入失败记录
            results.append({
                "function_name": "__init__",
                "code": code,
                "result": "failed"
            })
            return

        # ------------------- 动态执行并测试逻辑 -------------------
        exec_globals = {
            'Callable': Callable,
            'SOp': type('SOp', (object,), {}),
            'Value': type('Value', (object,), {}),
            'Map': type('Map', (object,), {}),
            'RASPExpr': type('RASPExpr', (object,), {}),
        }
        exec_locals = {}

        try:
            # 动态执行代码片段
            exec(code, exec_globals, exec_locals)

            # Create a dummy class to test the __init__ logic
            class DummyClass(exec_locals['Value']):
                def __init__(self):
                    self.inner_value = 1

            inner_instance = exec_globals['SOp']()
            f = lambda x: x

            # Create an instance of the main class, mimicking the conditions needed
            instance = exec_locals['__init__'](DummyClass(), f, inner_instance)

            # Test assertions
            self.assertIs(instance.inner, inner_instance, f"Inner attribute does not match expected SOp instance.")
            self.assertEqual(instance.f, f, "Function f not correctly assigned.")

            print("__init__ function: PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "__init__",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"__init__ function: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "__init__",
                "code": code,
                "result": "failed"
            })

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