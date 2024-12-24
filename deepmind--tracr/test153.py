import unittest
import json
import os
from typing import Any  # 确保注入的环境中有 Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCompiledModelFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[152]  # Get the 153rd JSON element
        if not cls.code_snippet:
            raise ValueError("Expected a code snippet in the 153rd JSON array position")

    def test_get_compiled_model(self):
        """Dynamically test the get_compiled_model function with additional checks."""
        results = []  # 收集要写入 JSONL 的测试结果
        code = self.code_snippet

        # ------------------- 静态检查代码片段 -------------------
        func_pattern = r"def\s+get_compiled_model\s*\("
        if "model.Transformer" not in code or not re.search(func_pattern, code):
            self.fail("Code snippet does not appear to have the correct function or Transformer usage.")
        
        # ------------------- 动态执行并测试逻辑 -------------------
        exec_globals = {
            'model': Any,  # Stub for model module
            '_make_embedding_modules': Any,  # Stub for _make_embedding_modules function
            'Any': Any,  # Ensure Any is available to the executed code
        }
        exec_locals = {}

        def mock_transformer(arg):
            return "MockTransformer"

        def mock_embedding_modules(residual_space, tokens_space, indices_space, output_space):
            class MockEmbedModules:
                token_embed = "MockTokenEmbed"
                pos_embed = "MockPosEmbed"
                unembed = "MockUnembed"
            return MockEmbedModules()

        exec_globals['model'] = type('MockModel', (object,), {
            'Transformer': mock_transformer,
            'CompiledTransformerModel': lambda transformer, token_embed, position_embed, unembed, use_unembed_argmax:
                                        {
                                            "transformer": transformer,
                                            "token_embed": token_embed,
                                            "position_embed": position_embed,
                                            "unembed": unembed,
                                            "use_unembed_argmax": use_unembed_argmax
                                        },
        })

        exec_globals['_make_embedding_modules'] = mock_embedding_modules

        try:
            # 动态执行代码片段
            exec(code, exec_globals, exec_locals)

            # 检查 get_compiled_model 是否真的存在
            if 'get_compiled_model' not in exec_locals:
                raise AssertionError("Function 'get_compiled_model' was not found in the executed locals.")

            # Invoke get_compiled_model to test its output
            compiled_model = exec_locals['get_compiled_model']()

            self.assertEqual(compiled_model['transformer'], "MockTransformer", "Transformer was not initialized correctly.")
            self.assertEqual(compiled_model['token_embed'], "MockTokenEmbed", "Token embedding was not initialized correctly.")
            self.assertEqual(compiled_model['position_embed'], "MockPosEmbed", "Position embedding was not initialized correctly.")
            self.assertEqual(compiled_model['unembed'], "MockUnembed", "Unembed component was not initialized correctly.")

            # If all assertions passed
            results.append({
                "function_name": "get_compiled_model",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            results.append({
                "function_name": "get_compiled_model",
                "code": code,
                "result": f"failed with error: {str(e)}"
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

        # 删除 function_name == "get_compiled_model" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "get_compiled_model"
        ]

        # 将新结果附加
        existing_records.extend(results)

        # 重写 test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()