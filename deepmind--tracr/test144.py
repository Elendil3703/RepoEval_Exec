import unittest
import json
import os
from typing import Sequence, Any

TEST_RESULT_JSONL = "test_result.jsonl"

class ConstantSOp:
    def __init__(self, value, check_length=False):
        self.value = value
        self.check_length = check_length

class Value:
    # Dummy class to represent a value, adaptable for test purposes
    pass

class TestEvalConstantSOp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[143]  # Get the 144th JSON element (index 143)
        if len(cls.code_snippet) < 1:
            raise ValueError("Expected at least one code snippet in the specified JSON array")

    def eval_constant_sop(self, sop: ConstantSOp, xs: Sequence[Value]) -> Sequence[Value]:
        if sop.check_length and (len(xs) != len(sop.value)):
            raise ValueError(
                f"Constant len {len(sop.value)} doesn't match input len {len(xs)}.")
        return sop.value

    def test_eval_constant_sop(self):
        """Test eval_constant_sop with various logical scenarios."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # 收集要写入 JSONL 的测试结果

        # Test cases
        test_cases = [
            # Case 1: Matching lengths, check_length=True
            (ConstantSOp([1, 2, 3], check_length=True), [Value(), Value(), Value()], 'passed'),
            # Case 2: Mismatched lengths, check_length=True
            (ConstantSOp([1, 2], check_length=True), [Value(), Value(), Value()], 'failed'),
            # Case 3: Mismatched lengths, check_length=False
            (ConstantSOp([1, 2], check_length=False), [Value(), Value(), Value()], 'passed'),
            # Case 4: Empty sequence, check_length=True
            (ConstantSOp([], check_length=True), [], 'passed'),
            # Case 5: Empty sequence, check_length=False
            (ConstantSOp([], check_length=False), [], 'passed'),
        ]

        for i, (sop, xs, expected) in enumerate(test_cases):
            with self.subTest(case_index=i):
                try:
                    result = self.eval_constant_sop(sop, xs)
                    # If no exception, validate the returned value
                    self.assertEqual(result, sop.value, "Returned value mismatch.")
                    if expected == 'failed':
                        raise AssertionError("Expected ValueError not raised.")
                    print(f"Test case {i}: PASSED\n")
                    passed_count += 1
                    results.append({
                        "function_name": "eval_constant_sop",
                        "case_index": i,
                        "result": "passed"
                    })
                except ValueError as e:
                    if expected == 'passed':
                        print(f"Test case {i}: FAILED, unexpected ValueError: {e}\n")
                        failed_count += 1
                        results.append({
                            "function_name": "eval_constant_sop",
                            "case_index": i,
                            "result": "failed"
                        })
                    else:
                        print(f"Test case {i}: PASSED, expected ValueError\n")
                        passed_count += 1
                        results.append({
                            "function_name": "eval_constant_sop",
                            "case_index": i,
                            "result": "passed"
                        })
                except Exception as e:
                    print(f"Test case {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "eval_constant_sop",
                        "case_index": i,
                        "result": "failed"
                    })

        # 最终统计信息
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(test_cases)}\n")
        self.assertEqual(passed_count + failed_count, len(test_cases), "Test count mismatch!")

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

        # 删除 function_name == "eval_constant_sop" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "eval_constant_sop"
        ]

        # 将新结果附加
        existing_records.extend(results)

        # 重写 test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()