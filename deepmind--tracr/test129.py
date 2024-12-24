import unittest
import json
import os
from typing import Callable

TEST_RESULT_JSONL = "test_result.jsonl"

class TestSequenceMapInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[128]  # Get the 129th JSON element

    def test_sequence_map_init(self):
        """Test the __init__ method of the SequenceMap class."""
        passed_count = 0
        failed_count = 0
        results = []

        # This snippet provides the context for the test
        code = self.code_snippet

        try:
            exec_globals = {
                'Callable': Callable,
                'logging': unittest.mock.MagicMock(),  # Mock logging
                'RASPExpr': unittest.mock.MagicMock(), # Mock RASPExpr
                'SOp': unittest.mock.MagicMock(name="SOp"), # Mock SOp
                'Value': unittest.mock.MagicMock(name="Value") # Mock Value
            }
            exec_locals = {}

            # Execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Get the SequenceMap class from executed code
            SequenceMap = exec_locals.get('SequenceMap')

            if SequenceMap is None:
                raise Exception("SequenceMap class not defined in the code snippet.")

            # Test inputs
            class MockSOp:
                pass
            
            class Mock_callable:
                def __call__(self, a, b):
                    return a + b

            try:
                # Test case where both SOp are different
                fst_op = MockSOp()
                snd_op = MockSOp()
                instance = SequenceMap(Mock_callable(), fst_op, snd_op)

                self.assertEqual(instance.f, Mock_callable())
                self.assertEqual(instance.fst, fst_op)
                self.assertEqual(instance.snd, snd_op)

                # Test case where both SOp are the same
                instance_same = SequenceMap(Mock_callable(), fst_op, fst_op)
                exec_globals['logging'].warning.assert_called_with(
                    "Creating a SequenceMap with both inputs being the same SOp is discouraged. You should use a Map instead."
                )

                # Assert if the types are checked correctly
                self.assertIsInstance(instance.fst, SOp)
                self.assertIsInstance(instance.snd, SOp)
                self.assertTrue(callable(instance.f))
                self.assertNotIsInstance(instance.f, RASPExpr)

                results.append({"function_name": "__init__", "code": code, "result": "passed"})
                passed_count += 1
            except Exception as e:
                results.append({"function_name": "__init__", "code": code, "result": "failed", "error": str(e)})
                failed_count += 1

        except Exception as e:
            results.append({"function_name": "__init__", "code": code, "result": "failed", "error": str(e)})
            failed_count += 1

        # Test Summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed.\n")
        assert passed_count + failed_count == 1, "Test count mismatch!"

        # ========= Write Test Results to test_result.jsonl =========
        # Read existing test_result.jsonl (ignore if it doesn't exist)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "__init__"
        existing_records = [rec for rec in existing_records if rec.get("function_name") != "__init__"]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()