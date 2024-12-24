import unittest
import json
import sys
import os
from typing import Sequence, Any  # 确保注入的环境中有 Sequence 和 Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGroundTruthInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[130]  # Get the 131st JSON element
        if len(cls.code_snippet) < 1:
            raise ValueError("Expected at least one code snippet in the JSON element")

    def test_init_method(self):
        """Dynamically test the __init__ method in the code snippet with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # 收集要写入 JSONL 的测试结果

        code = self.code_snippet
        with self.subTest(code_index=130):
            print(f"Running test for code snippet 130...")
            # ------------------- 静态检查 -------------------
            # 1) 确保 __init__ 方法存在
            if "def __init__(" not in code:
                print(f"Code snippet 130: FAILED, '__init__' function not found in code.\n")
                failed_count += 1
                # 写入失败记录
                results.append({
                    "function_name": "__init__",
                    "code": code,
                    "result": "failed"
                })
            else:
                # ------------------- 动态执行并测试逻辑 -------------------
                exec_globals = {
                    'Sequence': Sequence,
                    'Any': Any,  # 注入 Sequence 和 Any
                    '__name__': '__main__'
                }
                exec_locals = {}

                try:
                    # 动态执行代码片段
                    exec(code, exec_globals, exec_locals)

                    # 检查类是否已被定义
                    class_name = None
                    for var_name, var_value in exec_locals.items():
                        if isinstance(var_value, type):
                            class_name = var_name
                            break

                    if not class_name:
                        print(f"Code snippet 130: FAILED, no class with '__init__' method found.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "__init__",
                            "code": code,
                            "result": "failed"
                        })
                    else:
                        # 使用 __init__ 方法进行测试
                        TestClass = exec_locals[class_name]
                        instance = TestClass(["sample"], check_length=False)

                        # 验证属性 value 和 check_length
                        self.assertEqual(instance.value, ["sample"], 
                                         f"Code snippet 130: 'value' attribute not set correctly.")
                        self.assertEqual(instance.check_length, False, 
                                         f"Code snippet 130: 'check_length' attribute not set correctly.")

                        print(f"Code snippet 130: PASSED all assertions.\n")
                        passed_count += 1
                        results.append({
                            "function_name": "__init__",
                            "code": code,
                            "result": "passed"
                        })
                except Exception as e:
                    print(f"Code snippet 130: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })

        # 最终统计信息
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 1\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

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