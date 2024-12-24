import unittest
import json
import sys
import os
from typing import Any, Dict

TEST_RESULT_JSONL = "test_result.jsonl"

_DATAPIPELINE = {}

def register_class(cls, name):
    """Ground Truth Implementation for testing."""
    _DATAPIPELINE[name] = cls
    setattr(sys.modules[__name__], name, cls)
    return cls


class TestCarperAITrlxRegisterClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[1]
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the second JSON array")

    def _find_register_class(self, scope: Dict[str, Any]) -> Any:
        """
        递归查找 'register_class' 函数。
        若顶层没有找到，则会遍历所有子字典和可调用对象（函数），
        对于无参数的函数会尝试调用一次，以便可能在调用后将 register_class 注册到作用域中。
        """
        visited = set()
        queue = [scope]

        while queue:
            current = queue.pop()
            # 防止循环引用导致死递归
            if id(current) in visited:
                continue
            visited.add(id(current))

            # 如果是字典，就先直接在里面找 register_class
            if isinstance(current, dict):
                # 1) 直接检测字典的键
                for k, v in current.items():
                    if k == 'register_class' and callable(v):
                        return v

                # 2) 如果没找到，则把子字典或可调用的对象加到队列，继续BFS
                for k, v in current.items():
                    if isinstance(v, dict):
                        queue.append(v)
                    elif callable(v):
                        # 尝试调用无参函数，看会不会在调用后把 register_class 注入字典
                        if v.__code__.co_argcount == 0:
                            try:
                                v()
                            except Exception:
                                pass
                            # 调用完再次检测：有些函数可能在执行后才定义 register_class
                            found = self._find_register_class(current)
                            if found:
                                return found

            # 如果 scope 中的某个对象本身就是可调用对象（函数）
            elif callable(current):
                if current.__code__.co_argcount == 0:
                    # 调用后再查找
                    try:
                        current()
                    except Exception:
                        pass
                    found = self._find_register_class(scope)
                    if found:
                        return found

        return None

    def test_register_class_snippets(self):
        """Dynamically test all code snippets for 'register_class' in the JSON."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i} (register_class)...")

                # 动态执行环境
                exec_globals = {
                    'sys': sys,
                    '_DATAPIPELINE': {},  # 为每个 snippet 创建一个全新的 _DATAPIPELINE
                    'Any': Any
                }
                exec_globals['__name__'] = __name__  # 确保动态执行时引用当前模块
                exec_locals = exec_globals

                try:
                    exec(code, exec_globals, exec_locals)

                    # 查找 register_class 函数（支持嵌套函数）
                    snippet_register_class = self._find_register_class(exec_locals)
                    if not snippet_register_class:
                        print(f"Code snippet {i}: FAILED, 'register_class' not found after exec.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "register_class",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test the functionality of register_class
                    class SomeClass:
                        pass

                    snippet_register_class(SomeClass, "test_class")

                    # Check if _DATAPIPELINE contains the correct registration
                    _DATAPIPELINE_after_exec = exec_globals['_DATAPIPELINE']
                    self.assertIn(
                        "test_class",
                        _DATAPIPELINE_after_exec,
                        f"Snippet {i} failed: 'test_class' not found in _DATAPIPELINE"
                    )
                    self.assertEqual(
                        _DATAPIPELINE_after_exec["test_class"],
                        SomeClass,
                        f"Snippet {i} failed: _DATAPIPELINE['test_class'] != SomeClass"
                    )

                    # Ensure sys.modules[__name__] has the correct class set
                    if not hasattr(sys.modules[__name__], "test_class"):
                        raise AssertionError(f"Snippet {i} failed: 'test_class' was not set in sys.modules[__name__]")

                    self.assertIs(getattr(sys.modules[__name__], "test_class"), SomeClass,
                                  f"Snippet {i} failed: sys.modules[__name__].test_class is not SomeClass")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "register_class",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "register_class",
                        "code": code,
                        "result": "failed"
                    })

        # 打印总结
        total = len(self.code_snippets)
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {total}\n")
        self.assertEqual(passed_count + failed_count, total, "Test count mismatch!")

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

        # 删除 function_name == "register_class" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "register_class"
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