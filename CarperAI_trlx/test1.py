import unittest
import json
import sys
import re
import os
from typing import Any  # 确保注入的环境中有 Any

TEST_RESULT_JSONL = "test_result.jsonl"

_DATAPIPELINE = {}

def register_datapipeline(name):
    """Decorator used to register a CARP architecture
    Args:
        name: Name of the architecture
    """
    def register_class(cls, name):
        _DATAPIPELINE[name] = cls
        setattr(sys.modules[__name__], name, cls)
        return cls

    if isinstance(name, str):
        name = name.lower()
        return lambda c: register_class(c, name)

    cls = name
    name = cls.__name__
    register_class(cls, name.lower())

    return cls

class TestCarperAITrlxResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[0]  # Get the first JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the first JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # 收集要写入 JSONL 的测试结果

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # ------------------- 多一些逻辑检查 -------------------
                # 1) 静态检查：判断 snippet 中是否真的定义了 _DATAPIPELINE 以及 register_datapipeline
                if "_DATAPIPELINE" not in code:
                    print(f"Code snippet {i}: FAILED, '_DATAPIPELINE' not found in code.\n")
                    failed_count += 1
                    # 写入失败记录
                    results.append({
                        "function_name": "register_datapipeline",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                if "def register_datapipeline" not in code:
                    print(f"Code snippet {i}: FAILED, function 'register_datapipeline' not found.\n")
                    failed_count += 1
                    # 写入失败记录
                    results.append({
                        "function_name": "register_datapipeline",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                func_pattern = r"def\s+register_datapipeline\s*\("
                if not re.search(func_pattern, code):
                    print(f"Code snippet {i}: FAILED, incorrect signature for 'register_datapipeline'.\n")
                    failed_count += 1
                    # 写入失败记录
                    results.append({
                        "function_name": "register_datapipeline",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- 动态执行并测试逻辑 -------------------
                exec_globals = {
                    'sys': sys,
                    '_DATAPIPELINE': {},
                    'Any': Any,  # 注入 Any
                }
                exec_locals = {}

                try:
                    # 动态执行代码片段
                    exec(code, exec_globals, exec_locals)

                    # 检查 register_datapipeline 是否真的存在
                    if 'register_datapipeline' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'register_datapipeline' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "register_datapipeline",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # 使用 register_datapipeline 测试注册逻辑
                    @exec_locals['register_datapipeline']("test_pipeline")
                    class TestPipeline:
                        pass

                    # 再注册一次，看是否会报错或覆盖
                    @exec_locals['register_datapipeline']("test_pipeline")
                    class OverwritePipeline:
                        pass

                    # 获取执行后得到的 _DATAPIPELINE
                    _DATAPIPELINE_after_exec = exec_globals['_DATAPIPELINE']

                    # 测试：test_pipeline 应该已被注册为 OverwritePipeline
                    self.assertIn(
                        "test_pipeline",
                        _DATAPIPELINE_after_exec,
                        f"Code snippet {i} did not correctly register 'test_pipeline'.",
                    )
                    self.assertEqual(
                        _DATAPIPELINE_after_exec["test_pipeline"],
                        OverwritePipeline,
                        f"Code snippet {i} did not map 'test_pipeline' to the overwritten class.",
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "register_datapipeline",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "register_datapipeline",
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

        # 删除 function_name == "register_datapipeline" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "register_datapipeline"
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