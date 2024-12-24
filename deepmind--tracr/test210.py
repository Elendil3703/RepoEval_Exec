import unittest
import json
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class VectorInBasis:
    def __init__(self, magnitudes, space):
        self.magnitudes = magnitudes
        self.space = space

    def __eq__(self, other):
        return isinstance(other, VectorInBasis) and self.magnitudes == other.magnitudes and self.space == other.space

class Operator:
    def __init__(self, matrix, left_space, right_space):
        self.matrix = matrix
        self.left_space = left_space
        self.right_space = right_space

    def __call__(self, x: VectorInBasis, y: VectorInBasis) -> float:
        """Describes the action of the operator on vectors."""
        if x not in self.left_space:
            raise TypeError(f"x={x} not in self.left_space={self.left_space}.")
        if y not in self.right_space:
            raise TypeError(f"y={y} not in self.right_space={self.right_space}.")
        return (x.magnitudes.T @ self.matrix @ y.magnitudes).item()

class TestOperator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[209]  # Get the 210th JSON element

    def test_call_method(self):
        """Dynamically test the __call__ method of Operator class."""
        passed_count = 0
        failed_count = 0
        results = []

        # Mock data for testing
        left_space = {'v1', 'v2', 'v3'}
        right_space = {'u1', 'u2', 'u3'}
        matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        # Correct instances for testing
        x = VectorInBasis(magnitudes=[1, 2, 3], space=left_space)
        y = VectorInBasis(magnitudes=[1, 0, 0], space=right_space)

        operator = Operator(matrix=matrix, left_space=left_space, right_space=right_space)

        # Test case: successful operation
        try:
            result = operator(x, y)
            self.assertIsInstance(result, float, "Result should be a float.")
            passed_count += 1
            results.append({
                "function_name": "__call__",
                "result": "passed"
            })
        except Exception as e:
            failed_count += 1
            results.append({
                "function_name": "__call__",
                "result": "failed",
                "error": str(e)
            })

        # Test case: x not in left_space
        invalid_x = VectorInBasis(magnitudes=[4, 5, 6], space={'v4'})
        try:
            operator(invalid_x, y)
            failed_count += 1
            results.append({
                "function_name": "__call__",
                "result": "failed",
                "error": "Invalid x did not raise error."
            })
        except TypeError:
            passed_count += 1
            results.append({
                "function_name": "__call__",
                "result": "passed"
            })

        # Test case: y not in right_space
        invalid_y = VectorInBasis(magnitudes=[4, 5, 6], space={'u4'})
        try:
            operator(x, invalid_y)
            failed_count += 1
            results.append({
                "function_name": "__call__",
                "result": "failed",
                "error": "Invalid y did not raise error."
            })
        except TypeError:
            passed_count += 1
            results.append({
                "function_name": "__call__",
                "result": "passed"
            })

        # ============= 将测试结果写入 test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__call__"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()