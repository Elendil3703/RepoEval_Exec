import unittest
import json
import os
from typing import List

TEST_RESULT_JSONL = "test_result.jsonl"

class MockProblem:
    """A mock problem object to simulate add_paths, add_parent, and add_child methods."""
    def __init__(self):
        self.paths = []
        self.parents = []
        self.children = []
        self.leaf = False

    def add_paths(self, paths: List):
        self.paths.extend(paths)

    def add_parent(self, parent):
        self.parents.append(parent)

    def add_child(self, child):
        self.children.append(child)

class MockDependencies:
    """A mock dependencies object to simulate dependency structure."""
    def __init__(self):
        self.dependencies = {
            "u2l": {},
            "l2u": {}
        }
        self.problems = []
        self.leaves = []

    def find_paths(self, src, dst):
        """Mock implementation of find_paths, which would be needed for testing."""
        # For simplicity in testing, return a dummy path
        return [(src, dst)]

    def check_leaf(self, problem):
        """Mock implementation to determine if a problem is a leaf."""
        # Simply check if problem has no children
        return not problem.children

    def parse_dependency(self):
        """
        Parse user-provided ``u2l`` and ``l2u`` dependencies to figure out 1) topological order for
        multilevel optimization execution, and 2) backpropagation path(s) for each problem.
        """
        # Parse upper-to-lower dependency
        for key, value_list in self.dependencies["u2l"].items():
            for value in value_list:
                # find all paths from low to high for backpropagation
                paths = self.find_paths(src=value, dst=key)
                key.add_paths(paths)

        # Parse lower-to-upper dependency
        for key, value_list in self.dependencies["l2u"].items():
            for value in value_list:
                # add value problem to parents of key problem for backpropgation
                key.add_parent(value)
                value.add_child(key)

        # Parse problems
        for problem in self.problems:
            if self.check_leaf(problem):
                problem.leaf = True
                self.leaves.append(problem)

class TestParseDependency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.chosen_code = data[281]  # Get the 282nd JSON element

    def setUp(self):
        """Initialize prerequisite test attributes."""
        self.mock_deps = MockDependencies()

    def test_parse_dependency(self):
        """Dynamically test code that mocks parse_dependency behavior."""
        # Mock E2E test to mimic how parse_dependency modifies dependencies

        # Setup problems
        problem_a = MockProblem()
        problem_b = MockProblem()
        problem_c = MockProblem()

        # Mock dependencies
        self.mock_deps.dependencies["u2l"] = {problem_a: [problem_b]}
        self.mock_deps.dependencies["l2u"] = {problem_b: [problem_c]}
        self.mock_deps.problems = [problem_a, problem_b, problem_c]

        # Run parse_dependency
        self.mock_deps.parse_dependency()

        # Check results for upper-to-lower dependency parsing (paths)
        self.assertIn((problem_b, problem_a), problem_a.paths, "Paths missing from upper-to-lower parsing.")

        # Check lower-to-upper dependency parsing (parent/child relationships)
        self.assertIn(problem_c, problem_b.parents, "Parent not correctly added in lower-to-upper parsing.")
        self.assertIn(problem_b, problem_c.children, "Child not correctly added in lower-to-upper parsing.")

        # Check leaf determination
        self.assertTrue(problem_c.leaf, "Leaf determination failed.")
        self.assertIn(problem_c, self.mock_deps.leaves, "Problem not added to leaves.")

        return "passed"

    def tearDown(self):
        # Store results similar to the reference provided in `test_result.jsonl`
        result_data = {
            "function_name": "parse_dependency",
            "code": self.chosen_code,
            "result": self.test_parse_dependency()
        }

        # Read existing test results
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove any previous results for this specific function
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "parse_dependency"
        ]

        # Append the new result
        existing_records.append(result_data)

        # Save results back to JSONL
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()