import unittest
import json
import os
from typing import List

TEST_RESULT_JSONL = "test_result.jsonl"

class VectorInBasis:
    def __init__(self, basis_directions: List[str], magnitudes: List[float]):
        self.basis_directions = basis_directions
        self.magnitudes = magnitudes

    def __sub__(self, other: "VectorInBasis") -> "VectorInBasis":
        if self.basis_directions != other.basis_directions:
            raise TypeError(f"Subtracting incompatible bases: {self} - {other}")
        magnitudes = [a - b for a, b in zip(self.magnitudes, other.magnitudes)]
        return VectorInBasis(self.basis_directions, magnitudes)

class TestVectorInBasisSubtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[215]  # Get the 216th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON element")

    def test__sub__(self):
        """Test the __sub__ method of VectorInBasis class."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # 收集要写入 JSONL 的测试结果

        # Simulate some scenarios and validate the expected results
        try:
            basis1 = ["x", "y", "z"]
            v1 = VectorInBasis(basis1, [1.0, 2.0, 3.0])
            v2 = VectorInBasis(basis1, [0.5, 1.5, 2.5])
            expected_magnitudes = [0.5, 0.5, 0.5]
            
            result_vector = v1 - v2  # Using __sub__

            # Verify the result
            self.assertEqual(
                result_vector.magnitudes, expected_magnitudes,
                "__sub__ did not return expected magnitudes."
            )
            self.assertEqual(
                result_vector.basis_directions, v1.basis_directions,
                "Basis directions changed after subtraction."
            )

            print("Subtraction test: PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "__sub__",
                "code": self.code_snippets,
                "result": "passed"
            })
        except AssertionError as e:
            print(f"Subtraction test: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "__sub__",
                "code": self.code_snippets,
                "result": "failed"
            })

        # Test with incompatible bases
        try:
            basis2 = ["a", "b", "c"]
            v3 = VectorInBasis(basis2, [1.0, 2.0, 3.0])

            with self.assertRaises(TypeError):
                v1 - v3  # This should raise an exception

            print("Incompatible bases test: PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "__sub__",
                "code": self.code_snippets,
                "result": "passed"
            })
        except AssertionError as e:
            print(f"Incompatible bases test: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "__sub__",
                "code": self.code_snippets,
                "result": "failed"
            })

        # 最终统计信息
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {2}\n")

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

        # 删除 function_name == "__sub__" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__sub__"
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