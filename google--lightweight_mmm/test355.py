import unittest
import json
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class TestModelInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[354]  # Get the 355th code snippet

    def test_post_init(self):
        """Test the __post_init__ method for expected behaviors."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        code = self.code_snippet
        
        exec_globals = {
            '_NAMES_TO_MODEL_TRANSFORMS': {
                'hill_adstock': 'transform_hill_adstock',
                'adstock': 'transform_adstock',
                'carryover': 'transform_carryover'
            },
            '_MODEL_FUNCTION': 'some_model_function',
            'models': type('models', (object,), {
                'MODEL_PRIORS_NAMES': {'prior_a', 'prior_b'},
                'TRANSFORM_PRIORS_NAMES': {
                    'hill_adstock': {'prior_hill_a', 'prior_hill_b'},
                    'adstock': {'prior_ad_a', 'prior_ad_b'},
                    'carryover': {'prior_carry_a', 'prior_carry_b'}
                }
            })
        }
        
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if __post_init__ is correctly defined
            if '__post_init__' not in exec_locals:
                print(f"FAILED: '__post_init__' not found in exec_locals.\n")
                failed_count += 1
                results.append({
                    "function_name": "__post_init__",
                    "code": code,
                    "result": "failed"
                })
                return

            # Define a class to test __post_init__
            class TestClass:
                def __init__(self, model_name):
                    self.model_name = model_name
                    exec_locals['__post_init__'](self)

            # Test valid model_name
            try:
                obj = TestClass('hill_adstock')
                self.assertEqual(obj._model_function, 'some_model_function')
                self.assertEqual(obj._model_transform_function, 'transform_hill_adstock')
                self.assertEqual(obj._prior_names, {'prior_a', 'prior_b', 'prior_hill_a', 'prior_hill_b'})
                passed_count += 1
                results.append({
                    "function_name": "__post_init__",
                    "code": code,
                    "result": "passed"
                })
            except Exception as e:
                print(f"FAILED for valid model name 'hill_adstock' with error: {e}")
                failed_count += 1
                results.append({
                    "function_name": "__post_init__",
                    "code": code,
                    "result": "failed"
                })

            # Test invalid model_name
            try:
                TestClass('invalid_model')  # Should raise ValueError
                print("FAILED: ValueError not raised for invalid model name.\n")
                failed_count += 1
                results.append({
                    "function_name": "__post_init__",
                    "code": code,
                    "result": "failed"
                })
            except ValueError:
                passed_count += 1

        except Exception as e:
            print(f"FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "__post_init__",
                "code": code,
                "result": "failed"
            })

        # Test Summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed\n")
        
        # Write results to JSONL
        # Read existing test_result.jsonl (ignore if it does not exist)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "__post_init__"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__post_init__"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()