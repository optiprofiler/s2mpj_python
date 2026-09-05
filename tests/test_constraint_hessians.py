"""Exercise constraint transformations through the actual S2MPJ adapter."""

from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np

# Reuse the repository's core bootstrap before importing the adapter.
from test_s2mpj_python import s2mpj_load


def central_jacobian(function, x):
    step = 1e-6
    eye = np.eye(x.size)
    return np.column_stack([
        (function(x + step * direction) - function(x - step * direction))
        / (2 * step) for direction in eye
    ])


class MixedConstraints:
    """Upper, two-sided, equality, lower, and linear rows in S2MPJ order."""

    def __init__(self):
        self.n, self.m = 2, 5
        self.nle, self.neq, self.nge = 2, 1, 2
        self.lincons = np.array([4])
        self.x0 = np.array([0.7, -0.4])
        self.xlower = np.full(2, -np.inf)
        self.xupper = np.full(2, np.inf)
        self.clower = np.array([-np.inf, -3, 1, 2, -np.inf])
        self.cupper = np.array([4, 5, 1, np.inf, 7])
        self.hessians = np.array([
            [[2, 0], [0, 4]], [[0, 1], [1, 6]],
            [[8, 0], [0, 10]], [[12, 2], [2, 14]],
            [[0, 0], [0, 0]],
        ], dtype=float)
        self.linear = np.zeros((5, 2))
        self.linear[4] = [1, -2]

    def cx(self, x):
        return np.einsum('i,kij,j->k', x, self.hessians, x) / 2 + self.linear @ x

    def cJx(self, x):
        return self.cx(x), self.hessians @ x + self.linear

    def cJHx(self, x):
        return *self.cJx(x), list(self.hessians)


class ConstraintHessianTests(unittest.TestCase):
    def assert_derivatives(self, problem, x):
        np.testing.assert_allclose(
            problem.jcub(x), central_jacobian(problem.cub, x), atol=1e-7,
        )
        hessians = problem.hcub(x)
        self.assertEqual(len(hessians), len(problem.cub(x)))
        for index, hessian in enumerate(hessians):
            np.testing.assert_allclose(hessian, central_jacobian(
                lambda point: problem.jcub(point)[index], x,
            ), atol=1e-7)
        weights = np.arange(2, 2 + len(hessians), dtype=float)
        weights[1::2] *= -1
        np.testing.assert_allclose(
            np.einsum('k,kij->ij', weights, hessians),
            central_jacobian(lambda point: weights @ problem.jcub(point), x),
            atol=1e-7,
        )

    def test_hs13_lower_bound_against_jacobian_difference(self):
        problem = s2mpj_load('HS13')
        np.testing.assert_allclose(problem.hcub(problem.x0)[0], np.diag([-18, 0]))
        self.assert_derivatives(problem, problem.x0)

    def test_upper_lower_two_sided_and_multiplier_order(self):
        module = SimpleNamespace(ADAPTERCONSTRAINTS=MixedConstraints)
        with patch('s2mpj_tools.importlib.import_module', return_value=module):
            problem = s2mpj_load('ADAPTERCONSTRAINTS')
        raw = MixedConstraints()
        x = raw.x0
        c = raw.cx(x)
        np.testing.assert_allclose(problem.cub(x), [c[0]-4, c[1]-5, -c[1]-3, -c[3]+2])
        np.testing.assert_allclose(problem.hcub(x), [
            raw.hessians[0], raw.hessians[1], -raw.hessians[1], -raw.hessians[3],
        ])
        np.testing.assert_allclose(problem.hceq(x), [raw.hessians[2]])
        np.testing.assert_allclose(problem.aub, [[1, -2]])
        np.testing.assert_allclose(problem.bub, [7])
        self.assert_derivatives(problem, x)
        self.assert_derivatives(problem, x + [0.2, -0.1])

    def test_unconstrained_hessian_lists_stay_empty(self):
        problem = s2mpj_load('ROSENBR')
        self.assertEqual(len(problem.hcub(problem.x0)), 0)
        self.assertEqual(len(problem.hceq(problem.x0)), 0)


if __name__ == '__main__':
    unittest.main()
