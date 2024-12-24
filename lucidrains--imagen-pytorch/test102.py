import unittest
import json
import os

TEST_RESULT_JSONL = "test_result.jsonl"


class MockImagen:
    def __init__(self):
        self.only_train_unet_number = None


class MockTrainer:
    def __init__(self):
        self.only_train_unet_number = None
        self.imagen = MockImagen()

    def validate_unet_number(self, unet_number):
        # Simulate validation logic
        pass

    def wrap_unet(self, unet_number):
        # Simulate wrap logic
        pass

    def validate_and_set_unet_being_trained(self, unet_number=None):
        if self.exists(unet_number):
            self.validate_unet_number(unet_number)

        assert not self.exists(self.only_train_unet_number) or self.only_train_unet_number == unet_number, 'you cannot only train on one unet at a time. you will need to save the trainer into a checkpoint, and resume training on a new unet'

        self.only_train_unet_number = unet_number
        self.imagen.only_train_unet_number = unet_number

        if not self.exists(unet_number):
            return

        self.wrap_unet(unet_number)

    @staticmethod
    def exists(variable):
        return variable is not None


class TestValidateAndSetUnetBeingTrained(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[101]  # Get the 102th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 102th JSON array")

    def test_validate_and_set_unet_being_trained(self):
        """Test validate_and_set_unet_being_trained method."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []  # Collect results to write into JSONL
        trainer = MockTrainer()

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                exec_globals = {'MockTrainer': MockTrainer}
                exec_locals = {}

                try:
                    exec(code, exec_globals, exec_locals)

                    if 'validate_and_set_unet_being_trained' not in exec_locals:
                        raise ValueError("validate_and_set_unet_being_trained function not found in code snippet.")

                    # Set up test cases
                    unet_numbers = [None, 1, 2]
                    for unet_number in unet_numbers:
                        initial_unet_number = trainer.only_train_unet_number

                        # Call the function
                        exec_locals['validate_and_set_unet_being_trained'](trainer, unet_number)

                        # Validate the execution results
                        self.assertEqual(trainer.only_train_unet_number, unet_number,
                                         f"Code snippet {i} failed for unet_number {unet_number}.")
                        self.assertEqual(trainer.imagen.only_train_unet_number, unet_number,
                                         f"Code snippet {i} failed for imagen unet_number {unet_number}.")
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "validate_and_set_unet_being_trained",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                print(f"Code snippet {i}: PASSED all assertions.\n")
                passed_count += 1
                results.append({
                    "function_name": "validate_and_set_unet_being_trained",
                    "code": code,
                    "result": "passed"
                })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write results to test_result.jsonl =============
        # Read existing test_result.jsonl (ignore if it doesn't exist)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "validate_and_set_unet_being_trained"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "validate_and_set_unet_being_trained"
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