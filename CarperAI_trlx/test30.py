import unittest
import json
import sys
import re
import os
import gc
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCarperAITrlxPostInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[29]  # Get the 30th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 30th JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # 收集要写入 JSONL 的测试结果

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # ------------------- 多一些逻辑检查 -------------------
                # 1) 静态检查：判断 snippet 中是否真的定义了 post_init
                if "def post_init" not in code:
                    print(f"Code snippet {i}: FAILED, function 'post_init' not found.\n")
                    failed_count += 1
                    # 写入失败记录
                    results.append({
                        "function_name": "post_init",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                func_pattern = r"def\s+post_init\s*\("
                if not re.search(func_pattern, code):
                    print(f"Code snippet {i}: FAILED, incorrect signature for 'post_init'.\n")
                    failed_count += 1
                    # 写入失败记录
                    results.append({
                        "function_name": "post_init",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- 动态执行并测试逻辑 -------------------
                exec_globals = {
                    'sys': sys,
                    'gc': gc,
                    'Any': Any,
                }
                exec_locals = {}

                try:
                    # 动态执行代码片段
                    exec(code, exec_globals, exec_locals)

                    # 检查 post_init 是否真的存在
                    if 'post_init' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'post_init' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "post_init",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # 使用 post_init 测试逻辑
                    class MockValueHead:
                        def load_state_dict(self, state_dict, strict):
                            self.loaded_state_dict = state_dict
                            self.strict = strict

                    mock_v_head = MockValueHead()
                    exec_locals['self'] = type('MockModel', (object,), {'v_head': mock_v_head})()
                    
                    test_state_dict = {
                        "v_head.layer1.weight": [1, 2, 3],
                        "v_head.layer2.bias": [4, 5, 6],
                        "other_component.weight": [7, 8, 9],
                    }

                    exec_locals['post_init'](exec_locals['self'], test_state_dict)

                    # Ensure keys are transformed correctly
                    correct_keys = set([
                        "layer1.weight",
                        "layer2.bias",
                        "other_component.weight"
                    ])
                    
                    self.assertEqual(set(mock_v_head.loaded_state_dict.keys()), correct_keys,
                                     f"Code snippet {i} did not transform state_dict keys correctly.")
                    
                    # Check if original keys are removed and new keys are present
                    self.assertNotIn("v_head.layer1.weight", mock_v_head.loaded_state_dict,
                                     f"Code snippet {i} did not remove original key 'v_head.layer1.weight'.")
                    self.assertIn("layer1.weight", mock_v_head.loaded_state_dict,
                                  f"Code snippet {i} did not add new key 'layer1.weight'.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "post_init",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "post_init",
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

        # 删除 function_name == "post_init" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "post_init"
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