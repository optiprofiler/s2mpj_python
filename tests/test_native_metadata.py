"""Native metadata and load-time probes fail closed (audit items B5 and R30).

The fixture is a controlled native-compatible model, so these checks do not
claim that a shipped problem contains malformed metadata.
"""
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np

# Reuse the repository's core bootstrap before importing the adapter.
from test_s2mpj_python import s2mpj_load
import s2mpj_tools as adapter


class Native:
    """One nonlinear row; attributes are overridden per scenario."""
    overrides = {}

    def __init__(self):
        self.n, self.m = 1, 1
        self.x0 = np.array([0.0])
        self.xlower, self.xupper = np.array([-1e20]), np.array([1e20])
        self.clower, self.cupper = np.array([-1e20]), np.array([0.0])
        self.lincons = np.array([], dtype=int)
        self.nle, self.neq, self.nge = 1, 0, 0
        self.jacobian_calls = 0
        vars(self).update(self.overrides)

    def cJx(self, x):
        self.jacobian_calls += 1
        return self.cx(x), np.ones((1, 1))

    def cx(self, x):
        return np.array([x[0] + 1.0])

    def fx(self, x):
        return float(x @ x)


def load(cls=Native, **overrides):
    fixture = type('FIXTURE', (cls,), {'overrides': overrides})
    with patch.object(adapter, '_import_native_problem',
                      return_value=SimpleNamespace(FIXTURE=fixture)):
        return s2mpj_load('FIXTURE')


class BoundedNative(Native):
    """Two rows: one linear inequality followed by one nonlinear equality."""

    def __init__(self):
        super().__init__()
        self.n, self.m = 2, 2
        self.x0 = np.array([1.0, 2.0])
        self.xlower = np.full((2, 1), -1e20)
        self.xupper = np.full((2, 1), 1e20)
        self.clower = np.array([[-1e20], [0.0]])
        self.cupper = np.array([[3.0], [0.0]])
        self.lincons = np.array([0])
        self.nle, self.neq, self.nge = 1, 1, 0
        vars(self).update(self.overrides)

    def cx(self, x):
        return np.array([[x[0] + 2 * x[1] - 1.0], [x[0] ** 2 - x[1]]])

    def cJx(self, x):
        self.jacobian_calls += 1
        return self.cx(x), np.array([[1.0, 2.0], [2 * x[0], -1.0]])


class NativeMetadataTests(unittest.TestCase):
    def assert_rejected(self, text, **overrides):
        with self.assertRaisesRegex(ValueError, text):
            load(**overrides)

    def test_control_keeps_the_constraint(self):
        problem = load()
        self.assertEqual((problem.ptype, problem.mnlcon), ('n', 1))
        self.assertEqual(problem.maxcv(np.array([0.0])), 1.0)

    def test_nan_bounds_never_disappear(self):
        # A NaN upper bound used to yield an unconstrained problem with maxcv 0.
        self.assert_rejected('constraint upper bounds', cupper=np.array([np.nan]))
        self.assert_rejected('constraint lower bounds', clower=np.array([np.nan]))
        self.assert_rejected('variable lower bounds', xlower=np.array([np.nan]))
        self.assert_rejected('variable upper bounds', xupper=np.array([np.nan]))
        self.assert_rejected('initial point', x0=np.array([np.nan]))

    def test_shapes_are_checked_before_masks(self):
        self.assert_rejected('constraint upper bounds', cupper=np.array([0.0, 0.0]))
        self.assert_rejected('constraint lower bounds', clower=np.empty(0))
        self.assert_rejected('variable lower bounds', xlower=np.array([0.0, 1.0]))
        self.assert_rejected('initial point', x0=np.zeros((2, 2)))
        self.assert_rejected('not numeric', cupper=np.array(['upper']))
        # Native arrays are columns; a matrix with the same size is not.
        two = BoundedNative
        with self.assertRaisesRegex(ValueError, 'constraint upper bounds'):
            load(two, cupper=np.zeros((1, 2)), m=4, nle=4, neq=0, lincons=np.array([], dtype=int),
                 clower=np.zeros((4, 1)))

    def test_missing_bounds_with_declared_constraints(self):
        class Missing(Native):
            def __init__(self):
                super().__init__()
                del self.cupper
        with self.assertRaisesRegex(ValueError, 'without clower and cupper'):
            load(Missing)

    def test_impossible_bounds(self):
        self.assert_rejected('variable bounds', xlower=np.array([np.inf]))
        self.assert_rejected('variable bounds', xupper=np.array([-np.inf]))
        self.assert_rejected('constraint bounds', clower=np.array([np.inf]))
        self.assert_rejected('constraint bounds', cupper=np.array([-np.inf]))
        # The native sentinel means the same as infinity.
        self.assert_rejected('variable bounds', xlower=np.array([1e20]))
        self.assert_rejected('constraint bounds', cupper=np.array([-1e20]))
        # Crossed bounds cannot be satisfied.
        self.assert_rejected('variable bounds', xlower=np.array([2.0]), xupper=np.array([1.0]))
        self.assert_rejected('constraint bounds', clower=np.array([1.0]))

    def test_a_free_row_is_legitimate(self):
        problem = load(cupper=np.array([1e20]))
        self.assertEqual((problem.ptype, problem.mnlcon), ('u', 0))

    def test_equality_rows_need_equal_finite_bounds(self):
        for clower, cupper in ((-1e20, 1e20), (0.0, 1.0), (-1e20, 0.0)):
            with self.subTest(clower=clower, cupper=cupper):
                with self.assertRaisesRegex(ValueError, 'equality constraints'):
                    load(BoundedNative, clower=np.array([-1e20, clower]),
                         cupper=np.array([3.0, cupper]))
        with self.assertRaisesRegex(ValueError, 'equality constraints'):
            load(BoundedNative, nle=2, neq=1)

    def test_linear_conversion(self):
        problem = load(BoundedNative)
        np.testing.assert_array_equal(problem.aub, [[1.0, 2.0]])
        np.testing.assert_array_equal(problem.bub, [4.0])
        self.assertEqual((problem.mlcon, problem.mnlcon), (1, 1))
        self.assertEqual(problem.maxcv(np.array([1.0, 1.0])), 0.0)


class FailingJacobian(BoundedNative):
    def cJx(self, x):
        raise ArithmeticError('native Jacobian failed')


class NonlinearOnly(Native):
    """Legitimate zeroth-order model without any derivative."""

    def cJx(self, x):
        raise AssertionError('A nonlinear problem needs no Jacobian probe at load time.')


class LoadProbeTests(unittest.TestCase):
    def test_failed_linear_probe_raises_with_the_cause(self):
        # This used to fill the Jacobian with NaN and keep going.
        with self.assertRaisesRegex(RuntimeError, 'linear constraints') as caught:
            load(FailingJacobian)
        self.assertIsInstance(caught.exception.__cause__, ArithmeticError)

    def test_nonfinite_or_misshaped_linear_probe_raises(self):
        class NanValue(BoundedNative):
            def cJx(self, x):
                return np.array([[np.nan], [0.0]]), np.ones((2, 2))

        class NanJacobian(BoundedNative):
            def cJx(self, x):
                return np.zeros((2, 1)), np.array([[np.inf, 1.0], [0.0, 0.0]])

        class WrongShape(BoundedNative):
            def cJx(self, x):
                return np.zeros((2, 1)), np.ones((3, 2))

        for fixture in (NanValue, NanJacobian, WrongShape):
            with self.subTest(fixture=fixture.__name__):
                with self.assertRaisesRegex(RuntimeError, 'linear constraints'):
                    load(fixture)

    def test_nonlinear_row_values_do_not_invalidate_linear_rows(self):
        class NanNonlinearRow(BoundedNative):
            def cJx(self, x):
                return (np.array([[x[0] + 2 * x[1] - 1.0], [np.nan]]),
                        np.array([[1.0, 2.0], [np.nan, np.nan]]))

        problem = load(NanNonlinearRow)
        np.testing.assert_array_equal(problem.bub, [4.0])

    def test_problem_without_linear_rows_is_not_probed(self):
        problem = load(NonlinearOnly)
        self.assertEqual(problem.mnlcon, 1)
        self.assertEqual(problem.maxcv(np.array([0.0])), 1.0)


if __name__ == '__main__':
    unittest.main()
