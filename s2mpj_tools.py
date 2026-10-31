import builtins, hashlib, importlib.util, sys, os, io, re, threading
from collections.abc import Mapping
from contextlib import redirect_stdout
from functools import lru_cache
import numpy as np
import pandas as pd

from optiprofiler.opclasses import Problem
from optiprofiler.utils import get_logger, shorten_log_message


# Unmodified native problem files run `from s2mpjlib import *`, and historically
# lived under the global package name `python_problems`. Those names are shared
# by every S2MPJ checkout in a process, so the first import would answer for all
# of them. Native modules therefore use an identity owned by this installation,
# and neither sys.path nor the global module names are touched.
_NATIVE_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
_NATIVE_MODULE_PREFIX = '_optiprofiler_s2mpj_' + hashlib.sha256(
    os.path.realpath(_NATIVE_SRC_DIR).encode()
).hexdigest()[:16]
_NATIVE_IMPORT_LOCK = threading.RLock()


def _exec_native_file(module_name, path, module_builtins=None):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    if module_builtins is not None:
        # exec() honors a __builtins__ entry already present in the globals.
        module.__dict__['__builtins__'] = module_builtins
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


@lru_cache(maxsize=1)
def _native_builtins():
    """Return builtins resolving `s2mpjlib` to this provider's own helper."""
    helper_name = f'{_NATIVE_MODULE_PREFIX}_helper'
    helper = sys.modules.get(helper_name)
    if helper is None:
        helper = _exec_native_file(
            helper_name,
            os.path.join(_NATIVE_SRC_DIR, 's2mpjlib.py'),
        )

    def scoped_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 's2mpjlib' and level == 0:
            return helper
        return builtins.__import__(name, globals, locals, fromlist, level)

    return {**vars(builtins), '__import__': scoped_import}


def _import_native_problem(problem_name):
    """Import one unmodified native problem under the provider-owned identity."""
    path = os.path.join(_NATIVE_SRC_DIR, 'python_problems', f'{problem_name}.py')
    if not problem_name.isidentifier() or not os.path.isfile(path):
        raise ModuleNotFoundError(f"S2MPJ has no problem named '{problem_name}'.")
    module_name = f'{_NATIVE_MODULE_PREFIX}_problem_{problem_name}'
    with _NATIVE_IMPORT_LOCK:
        module = sys.modules.get(module_name)
        if module is None:
            module = _exec_native_file(module_name, path, _native_builtins())
    return module


def _native_vector(problem_name, label, values, size):
    """Return native metadata as a vector, rejecting a wrong shape or NaN."""
    if hasattr(values, 'toarray'):
        values = values.toarray()
    try:
        vector = np.atleast_1d(np.asarray(values, dtype=float))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid {label} for S2MPJ problem '{problem_name}': "
            'the values are not numeric.'
        ) from exc
    if vector.size != size or (size > 0 and max(vector.shape) != size):
        raise ValueError(
            f"Invalid {label} for S2MPJ problem '{problem_name}': expected "
            f'{size} entries, found shape {vector.shape}.'
        )
    if np.any(np.isnan(vector)):
        raise ValueError(
            f"Invalid {label} for S2MPJ problem '{problem_name}': NaN entries "
            'leave a bound undefined, and it must not silently disappear.'
        )
    return vector.reshape(-1)


def s2mpj_load(problem_name, *args, library_options=None):
    """
    Convert an S2MPJ problem name to a `Problem` instance.

    Parameters
    ----------
    problem_name : str
        Name of the problem in the S2MPJ collection. More details about
        S2MPJ can be found at
        `the official repository <https://github.com/GrattonToint/S2MPJ>`_.
    library_options : dict, optional
        Validated S2MPJ options supplied by OptiProfiler. They currently affect
        problem selection rather than loading, but accepting the same mapping
        keeps selector and worker behavior explicit.

    Returns
    -------
    optiprofiler.Problem
        A ``Problem`` instance corresponding to the named problem.

    Notes
    -----
    There are two ways to obtain the ``problem_name`` you want:

    1. Use `s2mpj_select` to get the problem names that satisfy your
       criteria.
    2. Look for the CSV file ``probinfo_python.csv`` in the same directory
       as this module. It contains information about all problems in S2MPJ.

    The problem name may appear in the form ``'PROBLEMNAME_n_m'`` where
    ``n`` is the dimension and ``m`` is the number of linear and nonlinear
    constraints. This happens when a problem accepts extra arguments to
    change its dimension or number of constraints. This information is
    stored in the ``probinfo_python.csv`` file.

    Invalid native metadata fails closed: wrong shapes, NaN entries and
    impossible or crossed bounds raise `ValueError`, and a linear constraint whose values
    or Jacobian cannot be read at ``x0`` raises `RuntimeError` with the
    original cause. No constraint is dropped and no NaN matrix is built.
    Problems without linear constraints need no Jacobian probe at load time.

    See Also
    --------
    s2mpj_select : Select problems from S2MPJ by criteria.

    Examples
    --------
    .. code-block:: python

        from optiprofiler.problem_libs.s2mpj import s2mpj_load

        problem = s2mpj_load('ROSENBR')
        print(problem.n)  # 2
    """

    if library_options is not None:
        s2mpj_validate_options(library_options)

    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Check if 'problem_name' has the pattern '_n_m' or '_n'. If it has, find the position of the pattern and return the dimension 'n' and the number of constraints 'm'.
    # Note that when 'm' is 0, the pattern is '_n' instead of '_n_0'.
    pattern = r'_(\d+)_(\d+)$|_(\d+)$'
    match = re.search(pattern, problem_name)
    if match:
        problem_name = problem_name[:match.start()]
        if match.group(1) and match.group(2):
            dim = int(match.group(1))
            mcon = int(match.group(2))
        elif match.group(3):
            dim = int(match.group(3))
            mcon = 0
        else:
            raise ValueError(f"Invalid problem name format: {problem_name}")
    else:
        dim = None
        mcon = None

    # Check csv file for the corresponding arguments.
    csv_file = os.path.join(current_dir, 'probinfo_python.csv')
    resolved = False
    if match:
        df = pd.read_csv(csv_file, index_col=0)
        if problem_name in df.index:
            row = df.loc[problem_name]
            if dim is not None:
                dims_str = str(row['dims']).strip()
                mcons_str = str(row['mcons']).strip()
                argins_str = str(row['argins']).strip()
                
                if dims_str and mcons_str and argins_str:
                    dims_list = [int(x) for x in dims_str.split()]
                    mcons_list = [int(x) for x in mcons_str.split()]
                    
                    if problem_name == 'NUFFIELD' and argins_str.startswith('{5.0}{'):
                        variable_part = argins_str[len('{5.0}{'):-1]
                        variable_args = [float(x) for x in variable_part.split()]
                        fixed_args = [5.0]
                    elif problem_name == 'TRAINF' and argins_str.startswith('{1.5}{2}{'):
                        variable_part = argins_str[len('{1.5}{2}{'):-1]
                        variable_args = [float(x) for x in variable_part.split()]
                        fixed_args = [1.5, 2]
                    else:
                        variable_args = [float(x) for x in argins_str.split()]
                        fixed_args = []
                    
                    matching_indices = []
                    for i, (d, m) in enumerate(zip(dims_list, mcons_list)):
                        if d == dim and m == mcon:
                            matching_indices.append(i)
                    
                    if matching_indices:
                        index = matching_indices[0]
                        if index < len(variable_args):
                            selected_arg = variable_args[index]

                            if fixed_args:
                                args = args + tuple(fixed_args) + (selected_arg,)
                            else:
                                args = args + (selected_arg,)
                            resolved = True
        if not resolved:
            raise ValueError(
                f"S2MPJ problem '{problem_name}' has no instance with dimension "
                f'{dim} and {mcon} constraints.'
            )

    # Import the problem module under the provider-owned identity.
    problem_module = _import_native_problem(problem_name)
    # Get the class from the module.
    problem_class = getattr(problem_module, problem_name)
    # Create an instance of the class from the module.
    p = problem_class(*args)

    # List of feasibility problems.
    feasibility_list = [
        'AIRCRFTA', 'ARGAUSS', 'ARGLALE', 'ARGLBLE', 'ARGTRIG', 'ARTIF', 'BAmL1SP', 'BARDNE', 'BEALENE', 'BENNETT5', 'BIGGS6NE', 'BOOTH', 'BOXBOD', 'BRATU2D', 'BRATU2DT', 'BRATU3D', 'BROWNBSNE', 'BROWNDENE', 'BROYDN3D', 'CBRATU2D', 'CBRATU3D', 'CHANDHEQ', 'CHEMRCTA', 'CHWIRUT2', 'CLUSTER', 'COOLHANS', 'CUBENE', 'CYCLIC3', 'CYCLOOCF', 'CYCLOOCT', 'DANIWOOD', 'DANWOOD', 'DECONVBNE', 'DENSCHNBNE', 'DENSCHNDNE', 'DENSCHNFNE', 'DEVGLA1NE', 'DEVGLA2NE', 'DRCAVTY1', 'DRCAVTY2', 'DRCAVTY3', 'ECKERLE4', 'EGGCRATENE', 'EIGENA', 'EIGENB', 'ELATVIDUNE', 'ENGVAL2NE', 'ENSO', 'ERRINROSNE', 'ERRINRSMNE', 'EXP2NE', 'EXTROSNBNE', 'FLOSP2HH', 'FLOSP2HL', 'FLOSP2HM', 'FLOSP2TH', 'FLOSP2TL', 'FLOSP2TM', 'FREURONE', 'GENROSEBNE', 'GOTTFR', 'GROWTH', 'GULFNE', 'HAHN1', 'HATFLDANE', 'HATFLDBNE', 'HATFLDCNE', 'HATFLDDNE', 'HATFLDENE', 'HATFLDFLNE', 'HATFLDF', 'HATFLDG', 'HELIXNE', 'HIMMELBA', 'HIMMELBC', 'HIMMELBD', 'HIMMELBFNE', 'HS1NE', 'HS25NE', 'HS2NE', 'HS8', 'HYDCAR20', 'HYDCAR6', 'HYPCIR', 'INTEGREQ', 'INTEQNE', 'KOEBHELBNE', 'KOWOSBNE', 'KSS', 'LANCZOS1', 'LANCZOS2', 'LANCZOS3', 'LEVYMONE10', 'LEVYMONE5', 'LEVYMONE6', 'LEVYMONE7', 'LEVYMONE8', 'LEVYMONE9', 'LEVYMONE', 'LIARWHDNE', 'LINVERSENE', 'LSC1', 'LSC2', 'LUKSAN11', 'LUKSAN12', 'LUKSAN13', 'LUKSAN14', 'LUKSAN17', 'LUKSAN21', 'LUKSAN22', 'MANCINONE', 'METHANB8', 'METHANL8', 'MEYER3NE', 'MGH09', 'MGH10', 'MISRA1A', 'MISRA1B', 'MISRA1C', 'MISRA1D', 'MODBEALENE', 'MSQRTA', 'MSQRTB', 'MUONSINE', 'n10FOLDTR', 'NELSON', 'NONSCOMPNE', 'NYSTROM5', 'OSBORNE1', 'OSBORNE2', 'OSCIGRNE', 'OSCIPANE', 'PALMER1ANE', 'PALMER1BNE', 'PALMER1ENE', 'PALMER1NE', 'PALMER2ANE', 'PALMER2BNE', 'PALMER2ENE', 'PALMER3ANE', 'PALMER3BNE', 'PALMER3ENE', 'PALMER4ANE', 'PALMER4BNE', 'PALMER4ENE', 'PALMER5ANE', 'PALMER5BNE', 'PALMER5ENE', 'PALMER6ANE', 'PALMER6ENE', 'PALMER7ANE', 'PALMER7ENE', 'PALMER8ANE', 'PALMER8ENE', 'PENLT1NE', 'PENLT2NE', 'POROUS1', 'POROUS2', 'POWELLBS', 'POWELLSQ', 'POWERSUMNE', 'PRICE3NE', 'PRICE4NE', 'QINGNE', 'QR3D', 'RAT42', 'RAT43', 'RECIPE', 'REPEAT', 'RES', 'ROSZMAN1', 'RSNBRNE', 'SANTA', 'SEMICN2U', 'SEMICON1', 'SEMICON2', 'SPECANNE', 'SSBRYBNDNE', 'SSINE', 'THURBER', 'TQUARTICNE', 'VANDERM1', 'VANDERM2', 'VANDERM3', 'VANDERM4', 'VARDIMNE', 'VESUVIA', 'VESUVIO', 'VESUVIOU', 'VIBRBEAMNE', 'WATSONNE', 'WAYSEA1NE', 'WAYSEA2NE', 'YATP1CNE', 'YATP2CNE', 'YFITNE', 'ZANGWIL3'
    ]
    is_feasibility = problem_name in feasibility_list

    # Collect the problem parameters for transforming into a Problem instance.
    name = problem_name

    fun = lambda x: _getfun(p, is_feasibility, name, x)
    grad = lambda x: _getgrad(p, is_feasibility, name, x)
    hess = lambda x: _gethess(p, is_feasibility, name, x)

    # Validate native metadata before building masks: a mask built on NaN
    # would silently drop the corresponding bound or constraint.
    n, m = p.n, getattr(p, 'm', 0)
    x0 = _native_vector(name, 'initial point', p.x0, n)
    xl = _native_vector(name, 'variable lower bounds', p.xlower, n)
    xu = _native_vector(name, 'variable upper bounds', p.xupper, n)
    if np.any(xl >= 1.0e+20) or np.any(xu <= -1.0e+20) or np.any(xl > xu):
        raise ValueError(f"Impossible variable bounds for S2MPJ problem '{name}'.")
    # We replace 1.0e+20 (which represents infinity) bounds with np.inf.
    xl = np.where(xl <= -1.0e+20, -np.inf, xl)
    xu = np.where(xu >= 1.0e+20, np.inf, xu)

    if m > 0 and not (hasattr(p, 'clower') and hasattr(p, 'cupper')):
        raise ValueError(
            f"Invalid constraint bounds for S2MPJ problem '{name}': it declares "
            f'{m} constraints without clower and cupper.'
        )
    cl = _native_vector(name, 'constraint lower bounds', p.clower, m) if m > 0 else np.empty(0)
    cu = _native_vector(name, 'constraint upper bounds', p.cupper, m) if m > 0 else np.empty(0)
    if np.any(cl >= 1.0e+20) or np.any(cu <= -1.0e+20) or np.any(cl > cu):
        raise ValueError(f"Impossible constraint bounds for S2MPJ problem '{name}'.")
    cl = np.where(cl <= -1.0e+20, -np.inf, cl)
    cu = np.where(cu >= 1.0e+20, np.inf, cu)
    idx_cl_finite = np.where(np.isfinite(cl))[0]
    idx_cu_finite = np.where(np.isfinite(cu))[0]

    if not hasattr(p, 'lincons'):
        p.lincons = np.array([], dtype=int)
    if not hasattr(p, 'nle'):
        p.nle = 0
    if not hasattr(p, 'neq'):
        p.neq = 0
    if not hasattr(p, 'nge'):
        p.nge = 0

    idx_native_eq = np.arange(p.nle, p.nle + p.neq)
    if np.any(idx_native_eq >= m) or np.any(cl[idx_native_eq] != cu[idx_native_eq]) \
            or not np.all(np.isfinite(cl[idx_native_eq])):
        raise ValueError(
            f"Invalid equality constraints for S2MPJ problem '{name}': each "
            'needs equal finite bounds.'
        )

    nonlincons = np.setdiff1d(np.arange(p.m), p.lincons)
    idx_eq = idx_native_eq
    idx_ineq = np.setdiff1d(np.arange(p.m), np.arange(p.nle, p.nle + p.neq))
    idx_le = np.intersect1d(idx_ineq, idx_cu_finite)
    idx_ge = np.intersect1d(idx_ineq, idx_cl_finite)
    idx_aeq = np.intersect1d(idx_eq, p.lincons)
    idx_aub_le = np.intersect1d(idx_le, p.lincons)
    idx_aub_ge = np.intersect1d(idx_ge, p.lincons)
    idx_cle = np.intersect1d(idx_le, nonlincons)
    idx_ceq = np.intersect1d(idx_eq, nonlincons)
    idx_cge = np.intersect1d(idx_ge, nonlincons)

    # The linear constraints are hidden in the cJx method output, cx = jx @ x - bx.
    # Read them only when needed, and fail closed: NaN matrices would describe
    # a less constrained problem, and a dropped row would be silently feasible.
    idx_linear = np.union1d(idx_aeq, np.union1d(idx_aub_le, idx_aub_ge))
    jx = bx = None
    if idx_linear.size > 0:
        try:
            with redirect_stdout(io.StringIO()):
                cx, jx = p.cJx(p.x0)[:2]
            cx = (cx.toarray() if hasattr(cx, 'toarray') else np.asarray(cx)).flatten()
            jx = jx.toarray() if hasattr(jx, 'toarray') else np.asarray(jx)
            if cx.shape != (m,) or jx.shape != (m, n):
                raise ValueError('Unexpected native constraint/Jacobian shape.')
            if not (np.all(np.isfinite(cx[idx_linear]))
                    and np.all(np.isfinite(jx[idx_linear, :]))):
                raise ValueError('Non-finite linear constraint/Jacobian values.')
            # A nonlinear row may be undefined at x0; only linear rows are used.
            with np.errstate(invalid='ignore', over='ignore'):
                bx = jx @ x0 - cx
            if not np.all(np.isfinite(bx[idx_linear])):
                raise ValueError('Non-finite linear constraint offsets.')
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load linear constraints for S2MPJ problem '{name}'."
            ) from exc

    # Note that in S2MPJ, the constraints are defined as:
    #  cl <= c(x) <= cu
    # Thus, the linear equality constraints are:
    #  jx[idx_aeq, :] @ x = bx[idx_aeq] + cu[idx_aeq]
    # and the linear inequality constraints are:
    #  jx[idx_aub_le, :] @ x <= bx[idx_aub_le] + cu[idx_aub_le]
    #  -jx[idx_aub_ge, :] @ x <= -bx[idx_aub_ge] - cl[idx_aub_ge]
    aeq = jx[idx_aeq, :] if idx_aeq.size > 0 else None
    aub = np.vstack([jx[idx_aub_le, :], -jx[idx_aub_ge, :]]) if (idx_aub_le.size > 0 or idx_aub_ge.size > 0) else None
    beq = bx[idx_aeq] + cu[idx_aeq] if idx_aeq.size > 0 else None
    bub = np.concatenate([bx[idx_aub_le] + cu[idx_aub_le], -bx[idx_aub_ge] - cl[idx_aub_ge]]) if (idx_aub_le.size > 0 or idx_aub_ge.size > 0) else None

    # Construct nonlinear constraint functions
    # Remind that in S2MPJ, the constraints are defined as:
    #  cl <= c(x) <= cu
    # Thus, the nonlinear equality constraints are:
    #  ceq(x) = c(x)[idx_ceq] - cu[idx_ceq] = 0
    # and the nonlinear inequality constraints are:
    #  cub(x) = [c(x)[idx_cle] - cu[idx_cle];
    #           -c(x)[idx_cge] + cl[idx_cge]] <= 0
    def ceq(x):
        if idx_ceq.size == 0: return None
        y = _getcx(p, name, x)
        if y is None: return np.full(idx_ceq.size, np.nan)
        z = y[idx_ceq] - cu[idx_ceq]
        return np.full(idx_ceq.size, np.nan) if (hasattr(z, "size") and z.size == 0) else z
    def cub(x):
        if idx_cle.size == 0 and idx_cge.size == 0: return None
        y = _getcx(p, name, x)
        if y is None: return np.full(idx_cle.size + idx_cge.size, np.nan)
        le = y[idx_cle] - cu[idx_cle] if idx_cle.size > 0 else None
        ge = y[idx_cge] - cl[idx_cge] if idx_cge.size > 0 else None
        if le is None and ge is None: return None
        if ge is None: return le
        if le is None: return -ge
        return np.concatenate([le, -ge])
    def hceq(x):
        if idx_ceq.size == 0: return []
        Hx = _getHx(p, name, x)
        if Hx is None: return [np.full((x.size, x.size), np.nan)] * idx_ceq.size
        return [Hx[i] for i in idx_ceq]
    def hcub(x):
        if idx_cle.size == 0 and idx_cge.size == 0: return []
        Hx = _getHx(p, name, x)
        if Hx is None: return [np.full((x.size, x.size), np.nan)] * (idx_cle.size + idx_cge.size)
        le = [Hx[i] for i in idx_cle] if idx_cle.size > 0 else []
        # Lower bounds use cl - c(x), so both derivative orders change sign.
        ge = [-Hx[i] for i in idx_cge] if idx_cge.size > 0 else []
        return le + ge

    def jceq(x):
        if idx_ceq.size == 0: return np.empty((0, p.n))
        Jx = _getJx(p, name, x)
        if Jx is None: return np.full((idx_ceq.size, x.size), np.nan)
        return Jx[idx_ceq, :]
    def jcub(x):
        if idx_cle.size == 0 and idx_cge.size == 0: return np.empty((0, p.n))
        Jx = _getJx(p, name, x)
        if Jx is None: return np.full((idx_cle.size + idx_cge.size, x.size), np.nan)
        le = Jx[idx_cle, :] if idx_cle.size > 0 else None
        ge = Jx[idx_cge, :] if idx_cge.size > 0 else None
        if le is None and ge is None: return np.empty((0, p.n))
        if ge is None: return le
        if le is None: return -ge
        return np.vstack([le, -ge])

    # Construct the Problem instance.
    problem = Problem(fun, x0, name=name, xl=xl, xu=xu, aub=aub, bub=bub, aeq=aeq, beq=beq, cub=cub, ceq=ceq, grad=grad, hess=hess, jcub=jcub, jceq=jceq, hcub=hcub, hceq=hceq)

    return problem


def s2mpj_collect_info():
    """Return the committed S2MPJ Python problem information table."""
    probinfo_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'probinfo_python.csv',
    )
    try:
        return pd.read_csv(probinfo_path).to_dict(orient='records')
    except Exception as exc:
        raise FileNotFoundError(
            f'Could not find or load problem info file at {probinfo_path}'
        ) from exc


def s2mpj_get_default_options():
    """Return raw package and environment-level option defaults.

    Validation is deliberately deferred until higher-priority process and
    per-run overrides have been merged by OptiProfiler.
    """
    config = {
        'variable_size': 'default',
        'test_feasibility_problems': 0,
    }
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'config.txt',
    )
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                for line in f:
                    stripped = line.strip()
                    if stripped.startswith('variable_size='):
                        config['variable_size'] = (
                            stripped.split('=', 1)[1]
                            .split('#', 1)[0]
                            .split('%', 1)[0]
                            .strip()
                        )
                    elif stripped.startswith('test_feasibility_problems='):
                        value = (
                            stripped.split('=', 1)[1]
                            .split('#', 1)[0]
                            .split('%', 1)[0]
                            .strip()
                        )
                        try:
                            value = int(value)
                        except ValueError:
                            pass
                        config['test_feasibility_problems'] = value
        except Exception:
            pass
    if 'S2MPJ_VARIABLE_SIZE' in os.environ:
        config['variable_size'] = os.environ['S2MPJ_VARIABLE_SIZE']
    if 'S2MPJ_TEST_FEASIBILITY_PROBLEMS' in os.environ:
        value = os.environ['S2MPJ_TEST_FEASIBILITY_PROBLEMS']
        try:
            value = int(value)
        except ValueError:
            pass
        config['test_feasibility_problems'] = value
    return config


def s2mpj_validate_options(options):
    """Validate and normalize S2MPJ-specific benchmark options."""
    if not isinstance(options, Mapping):
        raise TypeError('S2MPJ library options must be a mapping.')
    known = {'variable_size', 'test_feasibility_problems'}
    unknown = sorted(set(options) - known)
    if unknown:
        raise ValueError(
            f'Unknown S2MPJ library options: {unknown}. '
            f'Available options: {sorted(known)}.'
        )
    normalized = {
        'variable_size': options.get('variable_size', 'default'),
        'test_feasibility_problems': options.get(
            'test_feasibility_problems',
            0,
        ),
    }
    variable_size = normalized['variable_size']
    if not isinstance(variable_size, str) or variable_size not in {
        'default',
        'min',
        'max',
        'all',
    }:
        raise ValueError(
            "S2MPJ option `variable_size` must be 'default', 'min', 'max', or 'all'."
        )
    feasibility = normalized['test_feasibility_problems']
    if isinstance(feasibility, np.integer):
        feasibility = int(feasibility)
    if (
        isinstance(feasibility, bool)
        or not isinstance(feasibility, int)
        or feasibility not in {0, 1, 2}
    ):
        raise ValueError(
            'S2MPJ option `test_feasibility_problems` must be 0, 1, or 2.'
        )
    normalized['test_feasibility_problems'] = feasibility
    return normalized


def s2mpj_select(options, library_options=None):
    """
    Select problems from S2MPJ that satisfy given criteria.

    Parameters
    ----------
    options : dict
        A dictionary containing selection criteria. More details about
        S2MPJ can be found at
        `the official repository <https://github.com/GrattonToint/S2MPJ>`_.
        Supported keys:

        - **ptype** (*str*) -- Type of problems to select. A string
          consisting of any combination of ``'u'`` (unconstrained),
          ``'b'`` (bound constrained), ``'l'`` (linearly constrained),
          and ``'n'`` (nonlinearly constrained), such as ``'b'``,
          ``'ul'``, ``'ubn'``. Default is ``'ubln'``.
        - **mindim** (*int*) -- Finite minimum dimension. Default is ``1``.
        - **maxdim** (*int or positive infinity*) -- Maximum dimension.
          Default is ``float('inf')``.
        - **minb** (*int*) -- Minimum number of bound constraints.
          Default is ``0``.
        - **maxb** (*int or positive infinity*) -- Maximum number of bound constraints.
          Default is ``inf``.
        - **minlcon** (*int*) -- Minimum number of linear constraints.
          Default is ``0``.
        - **maxlcon** (*int or positive infinity*) -- Maximum number of linear constraints.
          Default is ``inf``.
        - **minnlcon** (*int*) -- Minimum number of nonlinear constraints.
          Default is ``0``.
        - **maxnlcon** (*int or positive infinity*) -- Maximum number of nonlinear constraints.
          Default is ``inf``.
        - **mincon** (*int*) -- Minimum total number of linear and
          nonlinear constraints. Default is ``0``.
        - **maxcon** (*int or positive infinity*) -- Maximum total number of linear and
          nonlinear constraints. Default is ``inf``.
        - **oracle** (*int*) -- Oracle provided by the problem. ``0``
          means zeroth-order, ``1`` means first-order, ``2`` means
          second-order. Default is ``0``.
        - **excludelist** (*list of str*) -- List of problem names to
          exclude. Default is ``[]``.
    library_options : dict, optional
        S2MPJ-specific options. Supported keys are ``variable_size`` and
        ``test_feasibility_problems``. If omitted, package configuration and
        environment-level defaults are used.

    Returns
    -------
    list of str
        Problem names that satisfy the given criteria.

    Notes
    -----
    Supply finite integer lower limits: ``mindim >= 1`` and the other
    ``min* >= 0``. Each upper limit is an integer at least its lower limit,
    or positive infinity (no upper cutoff). NaN and negative infinity are
    not valid selection limits. OptiProfiler validates this contract for
    ``benchmark`` calls; direct selector callers must supply valid criteria.

    1. All information about the problems can be found in the CSV file
       ``probinfo_python.csv`` in the same directory as this module.

    2. The problem name may appear in the form ``'PROBLEMNAME_n_m'``
       where ``n`` is the dimension and ``m`` is the number of
       constraints. This happens when a problem accepts extra arguments
       to change its dimension or number of constraints.

    3. There is a file ``config.txt`` in the same directory as this
       module. It can be used to set the options ``variable_size`` and
       ``test_feasibility_problems``. See the comments in ``config.txt``
       for details. You can also override these options at runtime using
       `set_plib_config` or by setting environment variables
       ``S2MPJ_VARIABLE_SIZE`` and ``S2MPJ_TEST_FEASIBILITY_PROBLEMS``.
       Environment variables take precedence over ``config.txt``. An explicit
       ``library_options`` mapping takes precedence over both and is used by
       ``benchmark(..., plib_options=...)``.

    See Also
    --------
    s2mpj_load : Load a problem from S2MPJ.
    optiprofiler.get_plib_config : Read the current configuration.
    optiprofiler.set_plib_config : Override configuration at runtime.

    Examples
    --------
    .. code-block:: python

        from optiprofiler.problem_libs.s2mpj import s2mpj_select

        names = s2mpj_select({'ptype': 'u', 'maxdim': 2})
    """
    if library_options is None:
        library_options = s2mpj_get_default_options()
    library_options = s2mpj_validate_options(dict(library_options))
    variable_size = library_options['variable_size']
    test_feasibility_problems = library_options['test_feasibility_problems']
    
    # Initialize result lists
    problem_names = []
    
    # Set default values for options
    default_options = {
        'ptype': 'ubln',
        'mindim': 1,
        'maxdim': np.inf,
        'minb': 0,
        'maxb': np.inf,
        'minlcon': 0,
        'maxlcon': np.inf,
        'minnlcon': 0, 
        'maxnlcon': np.inf,
        'mincon': 0,
        'maxcon': np.inf,
        'oracle': 0,
        'excludelist': []
    }
    for key in default_options:
        options.setdefault(key, default_options[key])
    
    # Add known problematic problems to exclude list
    exclude_problems = ['DANWOODLS', 'MISRA1CLS', 'ROSSIMP1', 'ROSSIMP2', 'ROSSIMP3']
    if options['oracle'] != 0:
        exclude_problems.append('NOZZLEfp')
    
    options['excludelist'] = list(set(options['excludelist'] + exclude_problems))
    
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load problem info from CSV file
    try:
        probinfo_path = os.path.join(current_dir, 'probinfo_python.csv')
        probinfo = pd.read_csv(probinfo_path)
    except:
        raise FileNotFoundError(f"Could not find or load problem info file at {probinfo_path}")
    
    # Helper function to safely convert values
    def safe_convert(value, default=0):
        if pd.isna(value) or value == 'unknown':
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            try:
                return float(value)
            except (ValueError, TypeError):
                return default
    
    # Process each problem
    for _, row in probinfo.iterrows():
        # Get problem default attributes
        problem_name = row['problem_name']
        ptype = row['ptype']
        is_feasibility = row['isfeasibility']
        dim = safe_convert(row['dim'])
        mb = safe_convert(row['mb'])
        mlcon = safe_convert(row['mlcon'])
        mnlcon = safe_convert(row['mnlcon'])
        mcon = safe_convert(row['mcon'])

        # Check if argins (variable sizes) exist
        argins = row['argins'] if pd.notna(row['argins']) else ''
        dims = row['dims'].split() if pd.notna(row['dims']) else []
        mbs = row['mbs'].split() if pd.notna(row['mbs']) else []
        mlcons = row['mlcons'].split() if pd.notna(row['mlcons']) else []
        mnlcons = row['mnlcons'].split() if pd.notna(row['mnlcons']) else []
        mcons = row['mcons'].split() if pd.notna(row['mcons']) else []
        if dims:
            dims = [safe_convert(d) for d in dims]
        if mbs:
            mbs = [safe_convert(m) for m in mbs]
        if mlcons:
            mlcons = [safe_convert(m) for m in mlcons]
        if mnlcons:
            mnlcons = [safe_convert(m) for m in mnlcons]
        if mcons:
            mcons = [safe_convert(m) for m in mcons]

        # Check if problem type matches
        if ptype not in options['ptype']:
            continue

        # Check feasibility problems based on user preference
        if test_feasibility_problems == 0:
            if is_feasibility:
                continue
        elif test_feasibility_problems == 1:
            if not is_feasibility:
                continue
        # If test_feasibility_problems == 2, include all problems
        
        # Check if default dimension and constraints satisfy criteria
        default_satisfy = (
            dim >= options['mindim'] and dim <= options['maxdim'] and
            mb >= options['minb'] and mb <= options['maxb'] and
            mlcon >= options['minlcon'] and mlcon <= options['maxlcon'] and
            mnlcon >= options['minnlcon'] and mnlcon <= options['maxnlcon'] and
            mcon >= options['mincon'] and mcon <= options['maxcon'] and
            problem_name not in options['excludelist']
        )
        
        # If default satisfies and (no variable sizes or we only want default), add the problem
        if default_satisfy and (not dims or variable_size == 'default'):
            problem_names.append(problem_name)
        
        # Skip variable size processing if we only want default
        if variable_size == 'default':
            continue
        
        # Process variable sizes if they exist
        if dims:
            # Create mask for configurations that satisfy criteria
            mask = []
            names = [f"{problem_name}_{dims[i]}_{mcons[i]}" if mcons[i] > 0 else f"{problem_name}_{dims[i]}" for i in range(len(dims))]
            for i in range(len(dims)):
                # Create the name for the current configuration
                name = names[i]
                satisfies = (
                    dims[i] >= options['mindim'] and dims[i] <= options['maxdim'] and
                    (i >= len(mbs) or mbs[i] >= options['minb']) and
                    (i >= len(mbs) or mbs[i] <= options['maxb']) and
                    (i >= len(mlcons) or mlcons[i] >= options['minlcon']) and
                    (i >= len(mlcons) or mlcons[i] <= options['maxlcon']) and
                    (i >= len(mnlcons) or mnlcons[i] >= options['minnlcon']) and
                    (i >= len(mnlcons) or mnlcons[i] <= options['maxnlcon']) and
                    (i >= len(mcons) or mcons[i] >= options['mincon']) and
                    (i >= len(mcons) or mcons[i] <= options['maxcon']) and
                    (dims[i] != dim or (i < len(mcons) and mcons[i] != mcon)) and
                    name not in options['excludelist']
                )
                mask.append(satisfies)
            
            idxs = [i for i, m in enumerate(mask) if m]
            
            if not idxs:
                if default_satisfy:
                    problem_names.append(problem_name)
                continue
            
            if variable_size == 'min':
                # Find minimum dimension among configurations that satisfy
                min_dim = min(dims[i] for i in idxs)
                idxs_dim = [i for i in idxs if dims[i] == min_dim]
                
                # Find minimum constraints among those
                if mcons and idxs_dim:
                    min_mcon = min(mcons[i] if i < len(mcons) else 0 for i in idxs_dim)
                    idxs = [i for i in idxs_dim if i >= len(mcons) or mcons[i] == min_mcon]
                    idxs = [idxs[0]]  # Take just the first
                    
                    # Compare with default
                    if default_satisfy and (dim < min_dim or (dim == min_dim and mcon < min_mcon)):
                        problem_names.append(problem_name)
                        continue
            
            elif variable_size == 'max':
                # Find maximum dimension among configurations that satisfy
                max_dim = max(dims[i] for i in idxs)
                idxs_dim = [i for i in idxs if dims[i] == max_dim]
                
                # Find maximum constraints among those
                if mcons and idxs_dim:
                    max_mcon = max(mcons[i] if i < len(mcons) else 0 for i in idxs_dim)
                    idxs = [i for i in idxs_dim if i < len(mcons) and mcons[i] == max_mcon]
                    idxs = [idxs[0]]  # Take just the first
                    
                    # Compare with default
                    if default_satisfy and (dim > max_dim or (dim == max_dim and mcon > max_mcon)):
                        problem_names.append(problem_name)
                        continue
            
            elif default_satisfy:
                # 'all' returns every satisfying instance, including the
                # default one (the variable-size rows above exclude it).
                problem_names.append(problem_name)
            
            # Add problems with specific dimensions/constraints
            for idx in idxs:
                if argins:
                    problem_names.append(names[idx])
    
    return problem_names

def _getfun(p, is_feasibility, problem_name, x):
    if is_feasibility and problem_name != 'HS8':
        # Note that 'HS8' has a constant objective function (-1).
        f = 0.0
    else:
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                f = p.fx(x)
                if hasattr(f, 'item'):
                    f = f.item()
            except Exception as err:
                _warn_s2mpj_evaluation_failure(problem_name, 'the objective function', err)
                f = np.nan
    return f

def _getgrad(p, is_feasibility, problem_name, x):
    if is_feasibility:
        g = np.zeros_like(x)
    else:
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                _, g = p.fgx(x)
                g = g.toarray() if hasattr(g, 'toarray') else g
                g = g.flatten()
            except Exception as err:
                _warn_s2mpj_evaluation_failure(problem_name, 'the gradient of the objective function', err)
                g = np.full(x.size, np.nan)
    return g

def _gethess(p, is_feasibility, problem_name, x):
    if is_feasibility:
        h = np.zeros((len(x), len(x)))
    else:
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                _, _, h = p.fgHx(x)
                h = h.toarray() if hasattr(h, 'toarray') else h
            except Exception as err:
                _warn_s2mpj_evaluation_failure(problem_name, 'the Hessian of the objective function', err)
                h = np.full((x.size, x.size), np.nan)
    return h

def _getcx(p, problem_name, x):
    if getattr(p, 'm', 0) == 0:
        return np.empty(0)
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            c = p.cx(x)
            c = c.toarray() if hasattr(c, 'toarray') else c
            c = c.flatten()
        except Exception as err:
            _warn_s2mpj_evaluation_failure(problem_name, 'the nonlinear constraint function', err)
            c = np.full(getattr(p, 'm', 0), np.nan)
    return c

def _getJx(p, problem_name, x):
    if getattr(p, 'm', 0) == 0:
        return np.empty((0, getattr(p, 'n', x.size)))
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            _, j = p.cJx(x)[:2]
            j = j.toarray() if hasattr(j, 'toarray') else j
        except Exception as err:
            _warn_s2mpj_evaluation_failure(problem_name, 'the Jacobian of the nonlinear constraint function', err)
            j = np.full((getattr(p, 'm', 0), getattr(p, 'n', x.size)), np.nan)
    return j

def _getHx(p, problem_name, x):
    if getattr(p, 'm', 0) == 0:
        return []
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            _, _, h = p.cJHx(x)
            for i in range(len(h)):
                if hasattr(h[i], 'toarray'):
                    h[i] = h[i].toarray()
        except Exception as err:
            _warn_s2mpj_evaluation_failure(problem_name, 'the Hessian of the nonlinear constraint function', err)
            h = [np.full((getattr(p, 'n', x.size), getattr(p, 'n', x.size)), np.nan) for _ in range(getattr(p, 'm', 0))]
        return h


def _warn_s2mpj_evaluation_failure(problem_name, what, err):
    logger = get_logger(__name__)
    msg = shorten_log_message(f'{type(err).__name__}: {err}')
    logger.warning(f'Failed to evaluate {what} of S2MPJ problem {problem_name}: {msg}')
