import unittest
import json
import os
from typing import Tuple, List, Any  # 确保使用到的类型注入

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCarperAIInitResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 读取JSON文件，选择第85组代码 (index 84)
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[84]
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 85th JSON array")

    def test_init_snippet(self):
        """Dynamically test all code snippets in the JSON related to __init__."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                # 检查代码中是否有 __init__ 函数及其他依赖
                if "def __init__(" not in code or "super().__init__()" not in code:
                    print(f"Code snippet {i}: FAILED, '__init__' function or 'super()' call not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # 动态执行并测试代码
                exec_globals = {
                    'Tuple': Tuple,
                    'List': List,
                    'Any': Any,
                    # 模拟 nn 和 Block 类
                    'nn': type('nn', (), {'ModuleList': list}), 
                    'Block': lambda x, y: (x, y),
                    'cast_tuple': lambda x, length=None: x if isinstance(x, tuple) else (x,) * (length or 1)
                }
                exec_locals = {}

                try:
                    exec(code, exec_globals, exec_locals)

                    # 检查目标类是否被定义
                    defined_classes = [cls for cls in exec_locals.values() if isinstance(cls, type)]
                    if not defined_classes:
                        print(f"Code snippet {i}: FAILED, target class not found.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "__init__",
                            "code": code,
                            "result": "failed"
                        })
                        continue
                    
                    # 假设我们只处理一个类，并创建其实例，进行具体测试
                    target_class = defined_classes[0]
                    
                    # Case 1: 测试 disabled 情况
                    test_instance = target_class(dim=128, enabled=False)
                    self.assertEqual(test_instance.dim_out, 128, f"Code snippet {i}: FAILED, dim_out should be 128 when disabled.")
                    
                    # Case 2: 测试 enabled 情况
                    test_instance = target_class(dim=128, enabled=True, dim_ins=(2, 4), dim_outs=(3, 3))
                    self.assertEqual(test_instance.dim_out, 128 + 6, f"Code snippet {i}: FAILED, dim_out should be 134 when enabled.")
                    self.assertEqual(len(test_instance.fmap_convs), 2, f"Code snippet {i}: FAILED, fmap_convs should have 2 elements when enabled.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })

        # 输出测试总结
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")

        # 将新测试结果写入 test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        existing_records.append(json.loads(line))

        # 删除 function_name == "__init__" 的旧记录
        existing_records = [rec for rec in existing_records if rec.get("function_name") != "__init__"]

        # 附加新结果
        existing_records.extend(results)

        # 写入到 test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()