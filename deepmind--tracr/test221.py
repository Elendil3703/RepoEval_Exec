import unittest
import json
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class VectorSpaceWithBasis:
    def __init__(self, basis):
        self.basis = basis

def direct_sum(*vs: VectorSpaceWithBasis) -> VectorSpaceWithBasis:
    total_basis = sum([v.basis for v in vs], [])

    if len(total_basis) != len(set(total_basis)):
        raise ValueError("Overlapping bases!")

    return VectorSpaceWithBasis(total_basis)

class TestDirectSumFunction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[220]  # Get the 221st JSON element

    def test_direct_sum(self):
        """Test the direct_sum function with various vectors."""
        passed_count = 0
        failed_count = 0
        results = []
        
        with self.subTest(code_index=220):
            try:
                # Test cases
                v1 = VectorSpaceWithBasis([1, 2, 3])
                v2 = VectorSpaceWithBasis([4, 5, 6])
                v3 = VectorSpaceWithBasis([7, 8, 9])

                # Case 1: No overlap, should pass
                result = direct_sum(v1, v2, v3)
                self.assertEqual(result.basis, [1, 2, 3, 4, 5, 6, 7, 8, 9])
                
                # Case 2: Overlapping bases, should fail
                v4 = VectorSpaceWithBasis([6, 7])
                with self.assertRaises(ValueError):
                    direct_sum(v1, v2, v4)

                print(f"Code snippet 220: PASSED all assertions.\n")
                passed_count += 1
                results.append({
                    "function_name": "direct_sum",
                    "code": self.code_snippet,
                    "result": "passed"
                })
            except Exception as e:
                print(f"Code snippet 220: FAILED with error: {e}\n")
                failed_count += 1
                results.append({
                    "function_name": "direct_sum",
                    "code": self.code_snippet,
                    "result": "failed"
                })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 1\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for "direct_sum"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "direct_sum"
        ]
        
        # Append new results
        existing_records.extend(results)

        # Re-write the file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()