"""Native S2MPJ modules keep a provider-owned identity in either import order.

Every check runs in a fresh interpreter because the subject is sys.modules and
sys.path state. The foreign fixture reuses the global names `python_problems`
and `s2mpjlib`, as another S2MPJ checkout in the same process would.
"""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[1]

FOREIGN_HELPER = """\
MARK = 'foreign helper'
class CUTEst_problem:
    def fx(self, x):
        return 999.0
"""
FOREIGN_PROBLEM = """\
from s2mpjlib import *
class ROSENBR(CUTEst_problem):
    n, m = 2, 0
"""
PRELUDE = """\
import importlib, os, sys
import numpy as np
foreign, root = sys.argv[1], sys.argv[2]
sys.path.insert(0, root)
import s2mpj_tools as adapter

def load_foreign():
    sys.path.insert(0, foreign)
    native = importlib.import_module('python_problems.ROSENBR')
    sys.path.remove(foreign)
    assert native.__file__.startswith(foreign), native.__file__
    assert sys.modules['s2mpjlib'].MARK == 'foreign helper'
    return native

def assert_bundled():
    problem = adapter.s2mpj_load('ROSENBR')
    assert problem.fun(np.array([1.0, 1.0])) == 0.0
    native = sys.modules[adapter._NATIVE_MODULE_PREFIX + '_problem_ROSENBR']
    assert native.__file__.startswith(root), native.__file__
    helper = native.ROSENBR.__mro__[1]
    assert helper.__name__ == 'CUTEst_problem' and helper is not sys.modules.get('s2mpjlib')
"""


class NativeIdentityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.foreign = Path(self.temporary.name) / 'foreign'
        (self.foreign / 'python_problems').mkdir(parents=True)
        (self.foreign / 's2mpjlib.py').write_text(FOREIGN_HELPER)
        (self.foreign / 'python_problems' / 'ROSENBR.py').write_text(FOREIGN_PROBLEM)

    def run_script(self, body, *arguments):
        script = PRELUDE + textwrap.dedent(body)
        completed = subprocess.run(
            [sys.executable, '-c', script, str(self.foreign), str(ROOT), *arguments],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_foreign_installation_imported_first(self):
        self.run_script("""
            native = load_foreign()
            assert_bundled()
            # The other installation stays intact and keeps its own values.
            assert sys.modules['python_problems.ROSENBR'] is native
            assert sys.modules['s2mpjlib'].MARK == 'foreign helper'
            assert native.ROSENBR().fx(None) == 999.0
        """)

    def test_foreign_installation_imported_second(self):
        self.run_script("""
            assert_bundled()
            native = load_foreign()
            assert native.ROSENBR().fx(None) == 999.0
            assert_bundled()
            # A first-time load after the foreign import is isolated as well.
            assert adapter.s2mpj_load('BEALE').fun(np.array([3.0, 0.5])) == 0.0
        """)

    def test_shared_helper_alone_is_not_reused(self):
        self.run_script("""
            sys.path.insert(0, foreign)
            import s2mpjlib
            sys.path.remove(foreign)
            assert s2mpjlib.MARK == 'foreign helper'
            assert_bundled()
            assert sys.modules['s2mpjlib'] is s2mpjlib
        """)

    def test_load_leaves_global_names_and_search_path_alone(self):
        self.run_script("""
            before = list(sys.path)
            assert_bundled()
            assert adapter.s2mpj_load('HS21').n == 2
            assert sys.path == before
            assert 'python_problems' not in sys.modules
            assert 's2mpjlib' not in sys.modules
            assert not any(name.startswith('python_problems.') for name in sys.modules)
        """)

    def test_two_installations_in_one_process_keep_their_own_problems(self):
        other = Path(self.temporary.name) / 'other'
        (other / 'src' / 'python_problems').mkdir(parents=True)
        for relative in ('s2mpj_tools.py', 'probinfo_python.csv', 'config.txt',
                         'src/s2mpjlib.py', 'src/python_problems/ROSENBR.py'):
            shutil.copyfile(ROOT / relative, other / relative)
        changed = other / 'src' / 'python_problems' / 'ROSENBR.py'
        source = changed.read_text()
        self.assertIn('self.gconst', source)
        # Shift the constant of the first group so the optimum value differs.
        marker = "name = 'ROSENBR'"
        self.assertIn(marker, source)
        changed.write_text(source.replace(marker, "name = 'ROSENBR'\n    def fx(self, x):\n        return 123.0"))
        self.run_script("""
            import importlib.util
            other = sys.argv[3]
            spec = importlib.util.spec_from_file_location('other_s2mpj_tools', other + '/s2mpj_tools.py')
            second = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(second)
            assert second._NATIVE_MODULE_PREFIX != adapter._NATIVE_MODULE_PREFIX
            x = np.array([1.0, 1.0])
            assert adapter.s2mpj_load('ROSENBR').fun(x) == 0.0
            assert second.s2mpj_load('ROSENBR').fun(x) == 123.0
            assert adapter.s2mpj_load('ROSENBR').fun(x) == 0.0
        """, str(other))

    def test_unknown_and_unsafe_names_are_rejected(self):
        self.run_script("""
            for name in ('NOSUCHPROBLEM', '../s2mpjlib', 'a.b'):
                try:
                    adapter._import_native_problem(name)
                except ModuleNotFoundError:
                    continue
                raise AssertionError(name)
        """)


if __name__ == '__main__':
    unittest.main()
