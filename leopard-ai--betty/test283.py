import unittest
import json
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class MockProblem:
    def __init__(self, name):
        self.name = name

class MockEnv:
    pass

class TestSetProblemAttr(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[282]  # Get the 283rd JSON element (index 282)

    def setUp(self):
        class MockEngine:
            def __init__(self, problems, env=None):
                self.problems = problems
                self.env = env

            def set_problem_attr(self, problem):
                """
                Set class attribute for the given ``problem`` based on their names

                :param problem: Problem in multilevel optimization
                :type problem: Problem
                :return: ``problem`` name
                :rtype: str
                """
                name = problem.name

                # set attribute for Engine
                assert not hasattr(self, name), f"Problem already has a problelm named {name}!"
                setattr(self, name, problem)

                # set attribute for Problems
                for prob in self.problems:
                    if prob != problem:
                        assert not hasattr(prob, name)
                        setattr(prob, name, problem)

                # set attribute for Env
                if self.env is not None:
                    setattr(self.env, name, problem)

                return name

        self.MockEngine = MockEngine

    def test_set_problem_attr(self):
        results = []

        code = self.code_snippet

        try:
            # Execute the code snippet in an isolated environment
            exec_globals = {}
            exec(code, exec_globals)

            # Load the set_problem_attr function
            set_problem_attr = exec_globals['set_problem_attr']

            # Test setup
            problem1 = MockProblem("Problem1")
            problem2 = MockProblem("Problem2")
            env = MockEnv()
            engine = self.MockEngine(problems=[problem1, problem2], env=env)

            # Test normal case
            name_set = set_problem_attr(engine, problem1)
            self.assertEqual(name_set, "Problem1")
            self.assertTrue(hasattr(engine, "Problem1"))
            self.assertTrue(hasattr(problem2, "Problem1"))
            self.assertTrue(hasattr(env, "Problem1"))

            # Test attribute conflict
            problem3 = MockProblem("Problem3")
            setattr(engine, "Problem3", problem3)
            with self.assertRaises(AssertionError):
                set_problem_attr(engine, problem3)

            print("Code snippet: PASSED all assertions.\n")
            results.append({
                "function_name": "set_problem_attr",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}\n")
            results.append({
                "function_name": "set_problem_attr",
                "code": code,
                "result": "failed"
            })

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for set_problem_attr
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "set_problem_attr"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite the file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()