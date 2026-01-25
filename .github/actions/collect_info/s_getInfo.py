"""
s_getInfo.py - Collect problem information from S2MPJ Python problem set

This script scans all problems in the S2MPJ Python collection and extracts
various metrics including dimensions, constraint counts, and function values.
The results are saved to CSV files for later use by OptiProfiler.

Usage: python s_getInfo.py
"""

import numpy as np
import pandas as pd
import signal
import os
import sys

# Get the repository root directory (three levels up from this script)
cwd = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(cwd, '..', '..', '..'))

# Add optiprofiler to the system path (checked out by GitHub Actions)
sys.path.append(os.path.join(repo_root, 'optiprofiler'))
sys.path.append(os.path.join(repo_root, 'optiprofiler', 'problems'))
from problems.s2mpj.s2mpj_tools import s2mpj_load

# Set the timeout (seconds) for each problem to be loaded
timeout = 50

# Read problem list from src directory
filename = os.path.join(repo_root, 'src', 'list_of_python_problems')
with open(filename, 'r') as file:
    problem_names = [line.strip().replace('.py', '') for line in file.readlines() 
                     if line.strip() and not line.startswith('#')]

# Exclude problematic problems that are known to cause issues
problem_exclude = [
    'SPARCO10LS', 'SPARCO10', 'SPARCO11LS', 'SPARCO11', 'SPARCO12LS', 'SPARCO12',
    'SPARCO2LS', 'SPARCO2', 'SPARCO3LS', 'SPARCO3', 'SPARCO5LS', 'SPARCO5',
    'SPARCO7LS', 'SPARCO7', 'SPARCO8LS', 'SPARCO8', 'SPARCO9LS', 'SPARCO9',
    'ROSSIMP3_mp', 'HS67', 'HS68', 'HS69', 'HS85', 'HS88', 'HS89', 'HS90',
    'HS91', 'HS92', 'TWIRIBG1'
]
problem_names = [name for name in problem_names if name not in problem_exclude]

# List of known feasibility problems (objective function is not meaningful)
known_feasibility = [
    'AIRCRFTA', 'ARGAUSS', 'ARGLALE', 'ARGLBLE', 'ARGTRIG', 'ARTIF', 'BAmL1SP',
    'BARDNE', 'BEALENE', 'BENNETT5', 'BIGGS6NE', 'BOOTH', 'BOXBOD', 'BRATU2D',
    'BRATU2DT', 'BRATU3D', 'BROWNBSNE', 'BROWNDENE', 'BROYDN3D', 'CBRATU2D',
    'CBRATU3D', 'CHANDHEQ', 'CHEMRCTA', 'CHWIRUT2', 'CLUSTER', 'COOLHANS',
    'CUBENE', 'CYCLIC3', 'CYCLOOCF', 'CYCLOOCT', 'DANIWOOD', 'DANWOOD',
    'DECONVBNE', 'DENSCHNBNE', 'DENSCHNDNE', 'DENSCHNFNE', 'DEVGLA1NE',
    'DEVGLA2NE', 'DRCAVTY1', 'DRCAVTY2', 'DRCAVTY3', 'ECKERLE4', 'EGGCRATENE',
    'EIGENA', 'EIGENB', 'ELATVIDUNE', 'ENGVAL2NE', 'ENSO', 'ERRINROSNE',
    'ERRINRSMNE', 'EXP2NE', 'EXTROSNBNE', 'FLOSP2HH', 'FLOSP2HL', 'FLOSP2HM',
    'FLOSP2TH', 'FLOSP2TL', 'FLOSP2TM', 'FREURONE', 'GENROSEBNE', 'GOTTFR',
    'GROWTH', 'GULFNE', 'HAHN1', 'HATFLDANE', 'HATFLDBNE', 'HATFLDCNE',
    'HATFLDDNE', 'HATFLDENE', 'HATFLDFLNE', 'HATFLDF', 'HATFLDG', 'HELIXNE',
    'HIMMELBA', 'HIMMELBC', 'HIMMELBD', 'HIMMELBFNE', 'HS1NE', 'HS25NE',
    'HS2NE', 'HS8', 'HYDCAR20', 'HYDCAR6', 'HYPCIR', 'INTEGREQ', 'INTEQNE',
    'KOEBHELBNE', 'KOWOSBNE', 'KSS', 'LANCZOS1', 'LANCZOS2', 'LANCZOS3',
    'LEVYMONE10', 'LEVYMONE5', 'LEVYMONE6', 'LEVYMONE7', 'LEVYMONE8',
    'LEVYMONE9', 'LEVYMONE', 'LIARWHDNE', 'LINVERSENE', 'LSC1', 'LSC2',
    'LUKSAN11', 'LUKSAN12', 'LUKSAN13', 'LUKSAN14', 'LUKSAN17', 'LUKSAN21',
    'LUKSAN22', 'MANCINONE', 'METHANB8', 'METHANL8', 'MEYER3NE', 'MGH09',
    'MGH10', 'MISRA1A', 'MISRA1B', 'MISRA1C', 'MISRA1D', 'MODBEALENE',
    'MSQRTA', 'MSQRTB', 'MUONSINE', 'n10FOLDTR', 'NELSON', 'NONSCOMPNE',
    'NYSTROM5', 'OSBORNE1', 'OSBORNE2', 'OSCIGRNE', 'OSCIPANE', 'PALMER1ANE',
    'PALMER1BNE', 'PALMER1ENE', 'PALMER1NE', 'PALMER2ANE', 'PALMER2BNE',
    'PALMER2ENE', 'PALMER3ANE', 'PALMER3BNE', 'PALMER3ENE', 'PALMER4ANE',
    'PALMER4BNE', 'PALMER4ENE', 'PALMER5ANE', 'PALMER5BNE', 'PALMER5ENE',
    'PALMER6ANE', 'PALMER6ENE', 'PALMER7ANE', 'PALMER7ENE', 'PALMER8ANE',
    'PALMER8ENE', 'PENLT1NE', 'PENLT2NE', 'POROUS1', 'POROUS2', 'POWELLBS',
    'POWELLSQ', 'POWERSUMNE', 'PRICE3NE', 'PRICE4NE', 'QINGNE', 'QR3D',
    'RAT42', 'RAT43', 'RECIPE', 'REPEAT', 'RES', 'ROSZMAN1', 'RSNBRNE',
    'SANTA', 'SEMICN2U', 'SEMICON1', 'SEMICON2', 'SPECANNE', 'SSBRYBNDNE',
    'SSINE', 'THURBER', 'TQUARTICNE', 'VANDERM1', 'VANDERM2', 'VANDERM3',
    'VANDERM4', 'VARDIMNE', 'VESUVIA', 'VESUVIO', 'VESUVIOU', 'VIBRBEAMNE',
    'WATSONNE', 'WAYSEA1NE', 'WAYSEA2NE', 'YATP1CNE', 'YATP2CNE', 'YFITNE',
    'ZANGWIL3'
]

# To store all the feasibility problems discovered during runtime
feasibility = []

# To store all the 'time out' problems
timeout_problems = []

# Output path (repository root)
saving_path = repo_root


class Logger:
    """Dual-output logger that writes to both terminal and log file."""
    def __init__(self, logfile):
        self.terminal = sys.__stdout__
        self.log = logfile

    def write(self, message):
        self.terminal.write(message)
        try:
            self.log.write(message)
        except Exception as e:
            self.terminal.write(f"[Logger Error] {e}\n")

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def run_with_timeout(func, args, timeout_seconds):
    """Execute a function with a timeout using SIGALRM."""
    def handler(signum, frame):
        raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_seconds)

    try:
        result = func(*args) if args else func()
        return result
    finally:
        signal.alarm(0)


def get_problem_info(problem_name, known_feasibility):
    """
    Extract information about a single problem.
    
    Returns a dictionary containing problem metrics such as dimensions,
    constraint counts, and function availability flags.
    """
    print(f"Processing problem: {problem_name}")

    info_single = {
        'problem_name': problem_name,
        'ptype': 'unknown',
        'xtype': 'unknown',
        'dim': 'unknown',
        'mb': 'unknown',
        'ml': 'unknown',
        'mu': 'unknown',
        'mcon': 'unknown',
        'mlcon': 'unknown',
        'mnlcon': 'unknown',
        'm_ub': 'unknown',
        'm_eq': 'unknown',
        'm_linear_ub': 'unknown',
        'm_linear_eq': 'unknown',
        'm_nonlinear_ub': 'unknown',
        'm_nonlinear_eq': 'unknown',
        'f0': 0,
        'isfeasibility': 1,
        'isgrad': 0,
        'ishess': 0,
        'isjcub': 0,
        'isjceq': 0,
        'ishcub': 0,
        'ishceq': 0
    }

    # Try to load the problem with timeout protection
    try:
        p = run_with_timeout(s2mpj_load, (problem_name,), timeout)
    except TimeoutError:
        print(f"Timeout while loading problem {problem_name}.")
        timeout_problems.append(problem_name)
        return info_single

    # Extract basic problem information
    try:
        info_single['ptype'] = p.ptype
        info_single['xtype'] = 'r'
        info_single['dim'] = p.n
        info_single['mb'] = p.mb
        info_single['ml'] = sum(p.xl > -np.inf)
        info_single['mu'] = sum(p.xu < np.inf)
        info_single['mcon'] = p.mcon
        info_single['mlcon'] = p.mlcon
        info_single['mnlcon'] = p.mnlcon
        info_single['m_ub'] = p.m_linear_ub + p.m_nonlinear_ub
        info_single['m_eq'] = p.m_linear_eq + p.m_nonlinear_eq
        info_single['m_linear_ub'] = p.m_linear_ub
        info_single['m_linear_eq'] = p.m_linear_eq
        info_single['m_nonlinear_ub'] = p.m_nonlinear_ub
        info_single['m_nonlinear_eq'] = p.m_nonlinear_eq
    except Exception as e:
        print(f"Error while getting problem info for {problem_name}: {e}")

    # Evaluate the objective function to determine if it's a feasibility problem
    try:
        f = run_with_timeout(p.fun, (p.x0,), timeout)
        if problem_name == 'LIN':
            info_single['isfeasibility'] = 0
            info_single['f0'] = np.nan
        elif np.size(f) == 0 or np.isnan(f) or problem_name in known_feasibility:
            info_single['isfeasibility'] = 1
            info_single['f0'] = 0
            feasibility.append(problem_name)
        else:
            info_single['isfeasibility'] = 0
            info_single['f0'] = f
    except Exception as e:
        print(f"Error while evaluating function for {problem_name}: {e}")
        info_single['f0'] = 0
        info_single['isfeasibility'] = 1
        feasibility.append(problem_name)

    # Check availability of gradient, Hessian, and constraint Jacobians/Hessians
    if problem_name in feasibility:
        info_single['isgrad'] = 1
        info_single['ishess'] = 1
    else:
        try:
            g = run_with_timeout(p.grad, (p.x0,), timeout)
            info_single['isgrad'] = 1 if g.size > 0 else 0
        except Exception:
            info_single['isgrad'] = 0

        try:
            h = run_with_timeout(p.hess, (p.x0,), timeout)
            info_single['ishess'] = 1 if h.size > 0 else 0
        except Exception:
            info_single['ishess'] = 0

    try:
        jc = run_with_timeout(p.jcub, (p.x0,), timeout)
        info_single['isjcub'] = 1 if jc.size > 0 else 0
    except Exception:
        info_single['isjcub'] = 0

    try:
        jc = run_with_timeout(p.jceq, (p.x0,), timeout)
        info_single['isjceq'] = 1 if jc.size > 0 else 0
    except Exception:
        info_single['isjceq'] = 0

    try:
        hc = run_with_timeout(p.hcub, (p.x0,), timeout)
        info_single['ishcub'] = 1 if len(hc) > 0 else 0
    except Exception:
        info_single['ishcub'] = 0

    try:
        hc = run_with_timeout(p.hceq, (p.x0,), timeout)
        info_single['ishceq'] = 1 if len(hc) > 0 else 0
    except Exception:
        info_single['ishceq'] = 0

    print(f"Finished processing problem {problem_name}.")
    return info_single


if __name__ == "__main__":
    # Set up logging to both terminal and file
    log_file = open(os.path.join(saving_path, 'log_python.txt'), 'w')
    sys.stdout = Logger(log_file)
    sys.stderr = Logger(log_file)

    # Process all problems and collect their information
    results = []
    for name in problem_names:
        info = get_problem_info(name, known_feasibility)
        results.append(info)
        sys.stdout.flush()
        sys.stderr.flush()

    # Create DataFrame and filter out problems with 'unknown' values
    df = pd.DataFrame(results)

    def has_unknown_values(row):
        for value in row:
            if str(value).strip().lower() == 'unknown':
                return True
        return False

    unknown_mask = df.apply(has_unknown_values, axis=1)
    if unknown_mask.any():
        filtered_problems = df.loc[unknown_mask, 'problem_name'].tolist()
        print(f"Filtered out {len(filtered_problems)} problems with 'unknown' values:")
        for problem in filtered_problems:
            print(f"  - {problem}")

    df_clean = df[~unknown_mask]

    # Save results to CSV
    df_clean.to_csv(os.path.join(saving_path, 'probinfo_python.csv'), index=False, na_rep='nan')

    # Save feasibility problems list
    with open(os.path.join(saving_path, 'feasibility_python.txt'), 'w') as f:
        f.write(' '.join(feasibility))

    # Save timeout problems list
    with open(os.path.join(saving_path, 'timeout_problems_python.txt'), 'w') as f:
        f.write(' '.join(timeout_problems))

    print("Script completed successfully.")

    # Clean up logging
    log_file.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
