from __future__ import annotations

from datetime import date
from pathlib import Path
from contextlib import contextmanager
import math
import os
import random
import sys
import unittest

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[1]

op_candidates = [
    REPO_DIR / "optiprofiler" / "python",
    REPO_DIR.parents[1] / "optiprofiler" / "python",
]
for op_path in op_candidates:
    if (op_path / "optiprofiler").is_dir():
        sys.path.insert(0, str(op_path))
        break

sys.path.insert(0, str(REPO_DIR))

from s2mpj_tools import s2mpj_load, s2mpj_select


REPRESENTATIVE_PROBLEMS = ["ALLINITU", "ALLINIT", "ALSOTAME", "ALLINITA"]


@contextmanager
def _temporary_env(**updates):
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _as_array(value):
    if value is None:
        return np.empty(0)
    return np.asarray(value)


def _assert_problem_contract(testcase, problem_name):
    problem = s2mpj_load(problem_name)
    testcase.assertEqual(problem.name.split("_")[0], problem_name.split("_")[0])
    testcase.assertGreaterEqual(problem.n, 1)
    testcase.assertEqual(problem.x0.size, problem.n)

    fx0 = problem.fun(problem.x0)
    testcase.assertTrue(math.isfinite(float(fx0)) or math.isnan(float(fx0)))

    cub0 = _as_array(problem.cub(problem.x0))
    ceq0 = _as_array(problem.ceq(problem.x0))
    testcase.assertEqual(cub0.ndim, 1)
    testcase.assertEqual(ceq0.ndim, 1)

    # Evaluate twice to catch wrappers that mutate S2MPJ state unexpectedly.
    fx1 = problem.fun(problem.x0)
    testcase.assertTrue(math.isfinite(float(fx1)) or math.isnan(float(fx1)))


class S2MPJPythonAdapterTests(unittest.TestCase):
    def test_select_representative_problem_types(self):
        selected = s2mpj_select(
            {
                "ptype": "ubln",
                "maxdim": 5,
                "maxb": 20,
                "maxlcon": 20,
                "maxnlcon": 20,
                "maxcon": 20,
            }
        )
        for problem_name in REPRESENTATIVE_PROBLEMS:
            self.assertIn(problem_name, selected)

    def test_load_representative_problem_types(self):
        for problem_name in REPRESENTATIVE_PROBLEMS:
            with self.subTest(problem=problem_name):
                _assert_problem_contract(self, problem_name)

    def test_daily_random_small_problem_sample(self):
        seed = int(os.environ.get("OP_RANDOM_SEED", date.today().strftime("%Y%m%d")))
        candidates = s2mpj_select(
            {
                "ptype": "ubln",
                "maxdim": 5,
                "maxb": 20,
                "maxlcon": 20,
                "maxnlcon": 20,
                "maxcon": 20,
            }
        )
        self.assertGreaterEqual(len(candidates), 4)

        rng = random.Random(seed)
        sample = rng.sample(candidates, k=min(4, len(candidates)))
        print(f"S2MPJ Python random sample seed={seed}: {sample}")
        for problem_name in sample:
            with self.subTest(problem=problem_name):
                _assert_problem_contract(self, problem_name)

    def test_config_environment_overrides_variable_size(self):
        options = {
            "ptype": "ubln",
            "maxdim": 5,
            "maxb": 20,
            "maxlcon": 20,
            "maxnlcon": 20,
            "maxcon": 20,
        }
        with _temporary_env(
            S2MPJ_VARIABLE_SIZE=None,
            S2MPJ_TEST_FEASIBILITY_PROBLEMS=None,
        ):
            default_names = s2mpj_select(dict(options))
        with _temporary_env(
            S2MPJ_VARIABLE_SIZE="all",
            S2MPJ_TEST_FEASIBILITY_PROBLEMS="0",
        ):
            all_names = s2mpj_select(dict(options))

        self.assertGreater(len(all_names), len(default_names))
        self.assertIn("CHEBYQAD_2", all_names)
        self.assertNotIn("CHEBYQAD_2", default_names)

    def test_config_environment_overrides_feasibility_selection(self):
        options = {
            "ptype": "ubln",
            "maxdim": 5,
            "maxb": 20,
            "maxlcon": 20,
            "maxnlcon": 20,
            "maxcon": 20,
        }
        with _temporary_env(
            S2MPJ_VARIABLE_SIZE="default",
            S2MPJ_TEST_FEASIBILITY_PROBLEMS="1",
        ):
            feasibility_names = s2mpj_select(dict(options))
        with _temporary_env(
            S2MPJ_VARIABLE_SIZE="default",
            S2MPJ_TEST_FEASIBILITY_PROBLEMS="2",
        ):
            all_names = s2mpj_select(dict(options))

        self.assertIn("ARGAUSS", feasibility_names)
        self.assertNotIn("ALLINITU", feasibility_names)
        self.assertIn("ARGAUSS", all_names)
        self.assertIn("ALLINITU", all_names)

    def test_config_environment_rejects_invalid_values(self):
        options = {"ptype": "u", "maxdim": 5}
        with _temporary_env(S2MPJ_VARIABLE_SIZE="not-a-mode"):
            with self.assertRaises(ValueError):
                s2mpj_select(dict(options))
        with _temporary_env(S2MPJ_TEST_FEASIBILITY_PROBLEMS="3"):
            with self.assertRaises(ValueError):
                s2mpj_select(dict(options))


if __name__ == "__main__":
    unittest.main()
