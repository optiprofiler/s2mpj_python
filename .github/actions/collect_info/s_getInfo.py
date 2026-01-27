import json
import subprocess
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from multiprocessing import Process, Queue
import signal
import shutil

# Add optiprofiler to the system path
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'optiprofiler'))

# Add problems to the system path
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cwd, 'optiprofiler', 'python'))
from optiprofiler.problem_libs.s2mpj.s2mpj_tools import s2mpj_load

# Set the timeout (seconds) for each problem to be loaded
timeout = 50

cwd = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(cwd, 'src', 'list_of_python_problems')
file = open(filename, 'r')
# Collect the names of the problems from the file
problem_names = [file.strip().replace('.py', '') for file in file.readlines() if file.strip() and not file.startswith('#')]
file.close()

# Exclude some problems
# 'HS67', 'HS68', 'HS69', 'HS85', 'HS88', 'HS89', 'HS90', 'HS91', 'HS92' are under development and not ready for use
# 'TWIRIBG1' will kill the process if run, so we exclude it
problem_exclude = [
    'SPARCO10LS', 'SPARCO10', 'SPARCO11LS', 'SPARCO11', 'SPARCO12LS', 'SPARCO12', 'SPARCO2LS', 'SPARCO2', 'SPARCO3LS', 'SPARCO3', 'SPARCO5LS', 'SPARCO5', 'SPARCO7LS', 'SPARCO7', 'SPARCO8LS', 'SPARCO8', 'SPARCO9LS', 'SPARCO9', 'ROSSIMP3_mp', 'HS67', 'HS68', 'HS69', 'HS85', 'HS88', 'HS89', 'HS90', 'HS91', 'HS92', 'TWIRIBG1'
]
problem_names = [name for name in problem_names if name not in problem_exclude]

# List all known feasibility problems
known_feasibility = [
    'AIRCRFTA', 'ARGAUSS', 'ARGLALE', 'ARGLBLE', 'ARGTRIG', 'ARTIF', 'BAmL1SP', 'BARDNE', 'BEALENE', 'BENNETT5', 'BIGGS6NE', 'BOOTH', 'BOXBOD', 'BRATU2D', 'BRATU2DT', 'BRATU3D', 'BROWNBSNE', 'BROWNDENE', 'BROYDN3D', 'CBRATU2D', 'CBRATU3D', 'CHANDHEQ', 'CHEMRCTA', 'CHWIRUT2', 'CLUSTER', 'COOLHANS', 'CUBENE', 'CYCLIC3', 'CYCLOOCF', 'CYCLOOCT', 'DANIWOOD', 'DANWOOD', 'DECONVBNE', 'DENSCHNBNE', 'DENSCHNDNE', 'DENSCHNFNE', 'DEVGLA1NE', 'DEVGLA2NE', 'DRCAVTY1', 'DRCAVTY2', 'DRCAVTY3', 'ECKERLE4', 'EGGCRATENE', 'EIGENA', 'EIGENB', 'ELATVIDUNE', 'ENGVAL2NE', 'ENSO', 'ERRINROSNE', 'ERRINRSMNE', 'EXP2NE', 'EXTROSNBNE', 'FLOSP2HH', 'FLOSP2HL', 'FLOSP2HM', 'FLOSP2TH', 'FLOSP2TL', 'FLOSP2TM', 'FREURONE', 'GENROSEBNE', 'GOTTFR', 'GROWTH', 'GULFNE', 'HAHN1', 'HATFLDANE', 'HATFLDBNE', 'HATFLDCNE', 'HATFLDDNE', 'HATFLDENE', 'HATFLDFLNE', 'HATFLDF', 'HATFLDG', 'HELIXNE', 'HIMMELBA', 'HIMMELBC', 'HIMMELBD', 'HIMMELBFNE', 'HS1NE', 'HS25NE', 'HS2NE', 'HS8', 'HYDCAR20', 'HYDCAR6', 'HYPCIR', 'INTEGREQ', 'INTEQNE', 'KOEBHELBNE', 'KOWOSBNE', 'KSS', 'LANCZOS1', 'LANCZOS2', 'LANCZOS3', 'LEVYMONE10', 'LEVYMONE5', 'LEVYMONE6', 'LEVYMONE7', 'LEVYMONE8', 'LEVYMONE9', 'LEVYMONE', 'LIARWHDNE', 'LINVERSENE', 'LSC1', 'LSC2', 'LUKSAN11', 'LUKSAN12', 'LUKSAN13', 'LUKSAN14', 'LUKSAN17', 'LUKSAN21', 'LUKSAN22', 'MANCINONE', 'METHANB8', 'METHANL8', 'MEYER3NE', 'MGH09', 'MGH10', 'MISRA1A', 'MISRA1B', 'MISRA1C', 'MISRA1D', 'MODBEALENE', 'MSQRTA', 'MSQRTB', 'MUONSINE', 'n10FOLDTR', 'NELSON', 'NONSCOMPNE', 'NYSTROM5', 'OSBORNE1', 'OSBORNE2', 'OSCIGRNE', 'OSCIPANE', 'PALMER1ANE', 'PALMER1BNE', 'PALMER1ENE', 'PALMER1NE', 'PALMER2ANE', 'PALMER2BNE', 'PALMER2ENE', 'PALMER3ANE', 'PALMER3BNE', 'PALMER3ENE', 'PALMER4ANE', 'PALMER4BNE', 'PALMER4ENE', 'PALMER5ANE', 'PALMER5BNE', 'PALMER5ENE', 'PALMER6ANE', 'PALMER6ENE', 'PALMER7ANE', 'PALMER7ENE', 'PALMER8ANE', 'PALMER8ENE', 'PENLT1NE', 'PENLT2NE', 'POROUS1', 'POROUS2', 'POWELLBS', 'POWELLSQ', 'POWERSUMNE', 'PRICE3NE', 'PRICE4NE', 'QINGNE', 'QR3D', 'RAT42', 'RAT43', 'RECIPE', 'REPEAT', 'RES', 'ROSZMAN1', 'RSNBRNE', 'SANTA', 'SEMICN2U', 'SEMICON1', 'SEMICON2', 'SPECANNE', 'SSBRYBNDNE', 'SSINE', 'THURBER', 'TQUARTICNE', 'VANDERM1', 'VANDERM2', 'VANDERM3', 'VANDERM4', 'VARDIMNE', 'VESUVIA', 'VESUVIO', 'VESUVIOU', 'VIBRBEAMNE', 'WATSONNE', 'WAYSEA1NE', 'WAYSEA2NE', 'YATP1CNE', 'YATP2CNE', 'YFITNE', 'ZANGWIL3'
]

# To store all the feasibility problems including the known ones and the new ones
feasibility = []

# To store all the 'time out' problems
timeout_problems = []

# Helper function to append to txt files
def append_to_txt(file_path, value):
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                existing = [line.strip() for line in f.readlines()]
            if value in existing:
                return
        with open(file_path, 'a') as f:
            f.write(value + '\n')
    except Exception as e:
        print(f"Error appending to {file_path}: {e}")


def _to_json_serializable(obj):
    """Convert numpy types in info_single to native Python for JSON."""
    if isinstance(obj, dict):
        return {k: _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# Find problems that are parametric
filename = os.path.join(cwd, 'list_of_parametric_problems_with_parameters_python.txt')
# Scan each line, each line only has one problem name, which ends before the first comma
# Give the rest to problem_argins
# In txt file, each line looks like:
# ALJAZZAF,3,100,1000,10000
# or
# TRAINF,{1.5}{2}{11,51,101,01,501,1001,5001,10001}
# ALJAZZAF and TRAINF are problem names
# Then let argins be the rest after the problem name if the problem name is found
with open(filename, 'r') as file:
    para_problem_names = []
    problem_argins = []
    for line in file:
        if line.strip() and not line.startswith('#'):
            parts = [x.strip() for x in line.split(',')]
            para_problem_names.append(parts[0])
            problem_argins.append(parts[1:])

saving_path = cwd

# Define the class logger
class Logger(object):
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

# Record the log from terminal
log_file = open(os.path.join(saving_path, 'log_python_temp.txt'), 'a')
sys.stdout = Logger(log_file)
sys.stderr = Logger(log_file)

def run_with_timeout(func, args, timeout_seconds):
    def handler(signum, frame):
        raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_seconds)
    
    try:
        result = func(*args) if args else func()
        return result
    finally:
        signal.alarm(0)

# Define a function to get information about a problem
def get_problem_info(problem_name, known_feasibility, problem_argins=None):

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
        'ishceq': 0,
        'argins': '',
        'dims': '',
        'mbs': '',
        'mls': '',
        'mus': '',
        'mcons': '',
        'mlcons': '',
        'mnlcons': '',
        'm_ubs': '',
        'm_eqs': '',
        'm_linear_ubs': '',
        'm_linear_eqs': '',
        'm_nonlinear_ubs': '',
        'm_nonlinear_eqs': '',
        'f0s': ''}
    try:
        p = run_with_timeout(s2mpj_load, (problem_name,), timeout)
    except TimeoutError:
        print(f"Timeout while loading problem {problem_name}.")
        timeout_problems.append(problem_name)
        append_to_txt(os.path.join(saving_path, 'timeout_problems_python_temp.txt'), problem_name)
        print(f"Skipping problem {problem_name} due to timeout.")
        return info_single

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

    try:
        f = run_with_timeout(p.fun, (p.x0,), timeout)
        if problem_name == 'LIN':
            info_single['isfeasibility'] = 0
        elif np.size(f) == 0 or np.isnan(f) or problem_name in known_feasibility:
            info_single['isfeasibility'] = 1
            feasibility.append(problem_name)
            append_to_txt(os.path.join(saving_path, 'feasibility_python_temp.txt'), problem_name)
        else:
            info_single['isfeasibility'] = 0
        if problem_name == 'LIN':
            info_single['f0'] = np.nan
        elif np.size(f) == 0 or np.isnan(f) or (problem_name in known_feasibility and problem_name != 'HS8'):
            info_single['f0'] = 0
        else:
            info_single['f0'] = f
    except Exception as e:
        print(f"Error while evaluating function for {problem_name}: {e}")
        info_single['f0'] = 0
        info_single['isfeasibility'] = 1
        feasibility.append(problem_name)
        append_to_txt(os.path.join(saving_path, 'feasibility_python_temp.txt'), problem_name)
    
    if problem_name in feasibility:
        info_single['isgrad'] = 1
        info_single['ishess'] = 1
    else:
        try:
            g = run_with_timeout(p.grad, (p.x0,), timeout)
            if g.size == 0:
                info_single['isgrad'] = 0
            else:
                info_single['isgrad'] = 1
        except Exception as e:
            print(f"Error while evaluating gradient for {problem_name}: {e}")
            info_single['isgrad'] = 0
        try:
            h = run_with_timeout(p.hess, (p.x0,), timeout)
            if h.size == 0:
                info_single['ishess'] = 0
            else:
                info_single['ishess'] = 1
        except Exception as e:
            print(f"Error while evaluating hessian for {problem_name}: {e}")
            info_single['ishess'] = 0
    
    try:
        jc = run_with_timeout(p.jcub, (p.x0,), timeout)
        if jc.size == 0:
            info_single['isjcub'] = 0
        else:
            info_single['isjcub'] = 1
    except Exception as e:
        print(f"Error while evaluating jcub for {problem_name}: {e}")
        info_single['isjcub'] = 0
    
    try:
        jc = run_with_timeout(p.jceq, (p.x0,), timeout)
        if jc.size == 0:
            info_single['isjceq'] = 0
        else:
            info_single['isjceq'] = 1
    except Exception as e:
        print(f"Error while evaluating jceq for {problem_name}: {e}")
        info_single['isjceq'] = 0
    
    try:
        hc = run_with_timeout(p.hcub, (p.x0,), timeout)
        if len(hc) == 0:
            info_single['ishcub'] = 0
        else:
            info_single['ishcub'] = 1
    except Exception as e:
        print(f"Error while evaluating hcub for {problem_name}: {e}")
        info_single['ishcub'] = 0
    
    try:
        hc = run_with_timeout(p.hceq, (p.x0,), timeout)
        if len(hc) == 0:
            info_single['ishceq'] = 0
        else:
            info_single['ishceq'] = 1
    except Exception as e:
        print(f"Error while evaluating hceq for {problem_name}: {e}")
        info_single['ishceq'] = 0

    if problem_argins is None:
        print(f"Finished processing problem {problem_name} without parameters.")
        return info_single

    # Collect additional information if the problem is parametric
    print(f"Processing parametric problem: {problem_name} with arguments {problem_argins}")
    # First handle two special cases:
    # NUFFIELD,{5.0}{10,20,30,40,100}
    # TRAINF,{1.5}{2}{11,51,101,201,501}
    if problem_name == 'NUFFIELD':
        fixed_argins = [5.0]
        variable_argins = [10, 20, 30, 40, 100]
    elif problem_name == 'TRAINF':
        fixed_argins = [1.5, 2]
        variable_argins = [11, 51, 101, 201, 501]
    else:
        fixed_argins = []
        variable_argins = problem_argins

    # Define a sub-function to process each argument (so that later we can use the ``run_with_timeout`` function)
    def process_arg(problem_name, arg, fixed_argins):
        try:
            p = s2mpj_load(problem_name, *fixed_argins, arg)

            result = {}
            result['n'] = p.n
            result['mb'] = p.mb
            result['ml'] = sum(p.xl > -np.inf)
            result['mu'] = sum(p.xu < np.inf)

            try:
                result['mcon'] = p.mcon
            except AttributeError as e:
                if "'Problem' object has no attribute '_m_nonlinear_ub'" in str(e):
                    result['mcon'] = p.mlcon + p.m_nonlinear_ub + p.m_nonlinear_eq
                else:
                    raise e
            
            result['mlcon'] = p.mlcon
            
            try:
                result['mnlcon'] = p.mnlcon
            except AttributeError as e:
                if "'Problem' object has no attribute '_m_nonlinear" in str(e):
                    result['mnlcon'] = p.m_nonlinear_ub + p.m_nonlinear_eq
                else:
                    raise e
            
            result['m_ub'] = p.m_linear_ub + p.m_nonlinear_ub
            result['m_eq'] = p.m_linear_eq + p.m_nonlinear_eq
            result['m_linear_ub'] = p.m_linear_ub
            result['m_linear_eq'] = p.m_linear_eq
            result['m_nonlinear_ub'] = p.m_nonlinear_ub
            result['m_nonlinear_eq'] = p.m_nonlinear_eq
            
            if problem_name in known_feasibility:
                result['f0'] = 0
            else:
                f = p.fun(p.x0)
                if np.size(f) == 0 or np.isnan(f):
                    result['f0'] = 0
                else:
                    result['f0'] = f
                    
            return True, arg, result
        except Exception as e:
            print(f"Error processing argument {arg} for problem {problem_name}: {e}")
            return False, arg, None

    successful_args = []
    for arg in variable_argins:
        print(f"Processing argument: {arg} for problem: {problem_name}")
        try:
            success, processed_arg, result = run_with_timeout(process_arg, (problem_name, arg, fixed_argins), timeout)
            if not success or result is None:
                print(f"Failed to process argument {arg} for problem {problem_name}")
                continue

            successful_args.append(processed_arg)
            info_single['dims'] += str(result['n']) + ' '
            info_single['mbs'] += str(result['mb']) + ' '
            info_single['mls'] += str(result['ml']) + ' '
            info_single['mus'] += str(result['mu']) + ' '
            info_single['mcons'] += str(result['mcon']) + ' '
            info_single['mlcons'] += str(result['mlcon']) + ' '
            info_single['mnlcons'] += str(result['mnlcon']) + ' '
            info_single['m_ubs'] += str(result['m_ub']) + ' '
            info_single['m_eqs'] += str(result['m_eq']) + ' '
            info_single['m_linear_ubs'] += str(result['m_linear_ub']) + ' '
            info_single['m_linear_eqs'] += str(result['m_linear_eq']) + ' '
            info_single['m_nonlinear_ubs'] += str(result['m_nonlinear_ub']) + ' '
            info_single['m_nonlinear_eqs'] += str(result['m_nonlinear_eq']) + ' '
            info_single['f0s'] += str(result['f0']) + ' '
        except TimeoutError:
            print(f"Timeout while processing problem {problem_name} with argument {arg}.")
            timeout_problems.append(problem_name + f" with arg {arg}")
            append_to_txt(os.path.join(saving_path, 'timeout_problems_python_temp.txt'), problem_name + f" with arg {arg}")
            continue
        except Exception as e:
            print(f"Error while processing problem {problem_name} with argument {arg}: {e}")
            continue

    if fixed_argins:
        info_single['argins'] = ''.join(['{' + str(fa) + '}' for fa in fixed_argins])
        info_single['argins'] += '{' + ' '.join(str(arg) for arg in successful_args) + '}'
    else:
        info_single['argins'] = ' '.join(str(arg) for arg in successful_args)

    info_single['dims'] = info_single['dims'].strip()
    info_single['mbs'] = info_single['mbs'].strip()
    info_single['mls'] = info_single['mls'].strip()
    info_single['mus'] = info_single['mus'].strip()
    info_single['mcons'] = info_single['mcons'].strip()
    info_single['mlcons'] = info_single['mlcons'].strip()
    info_single['mnlcons'] = info_single['mnlcons'].strip()
    info_single['m_ubs'] = info_single['m_ubs'].strip()
    info_single['m_eqs'] = info_single['m_eqs'].strip()
    info_single['m_linear_ubs'] = info_single['m_linear_ubs'].strip()
    info_single['m_linear_eqs'] = info_single['m_linear_eqs'].strip()
    info_single['m_nonlinear_ubs'] = info_single['m_nonlinear_ubs'].strip()
    info_single['m_nonlinear_eqs'] = info_single['m_nonlinear_eqs'].strip()
    info_single['f0s'] = info_single['f0s'].strip()

    print(f"Finished processing problem {problem_name} with parameters.")
    return info_single

if __name__ == "__main__":
    # --single mode: run one problem in subprocess isolation. Crashes (segfault/OOM) only kill
    # the child; parent excludes the problem and continues. Output written to result_single.json.
    if len(sys.argv) >= 2 and sys.argv[1] == "--single":
        try:
            name = sys.argv[2]
            args = json.loads(sys.argv[3]) if len(sys.argv) > 3 else None
            info = get_problem_info(name, known_feasibility, args)
            out = os.path.join(saving_path, "result_single.json")
            with open(out, "w") as f:
                json.dump(_to_json_serializable(info), f, indent=None)
            sys.exit(0)
        except Exception as e:
            print(f"[--single] Error processing {sys.argv[2] if len(sys.argv) > 2 else '?'}: {e}")
            sys.exit(1)

    csv_file = os.path.join(saving_path, 'probinfo_python.csv')
    csv_file_temp = os.path.join(saving_path, 'probinfo_python_temp.csv')
    current_prob_file = os.path.join(saving_path, 'current_problem.txt')
    exclude_file = os.path.join(saving_path, 'exclude_python.txt')
    result_single_path = os.path.join(saving_path, 'result_single.json')

    # 1. Crash Detection: If current_problem.txt exists, the previous run crashed on that problem.
    if os.path.exists(current_prob_file):
        with open(current_prob_file, 'r') as f:
            crashed_prob = f.read().strip()
        if crashed_prob:
            print(f"Detected crash during previous run on problem: {crashed_prob}. Adding to exclude list.")
            with open(exclude_file, 'a') as f:
                f.write(crashed_prob + '\n')
        try:
            os.remove(current_prob_file)
        except:
            pass

    # 2. Load existing exclusions
    if os.path.exists(exclude_file):
        with open(exclude_file, 'r') as f:
            for line in f:
                ex = line.strip()
                if ex and ex not in problem_exclude:
                    problem_exclude.append(ex)
    
    # Update problem_names by removing excluded ones
    problem_names = [name for name in problem_names if name not in problem_exclude]

    # 3. Resume: Find which problems are already completed
    completed_problems = set()
    if os.path.exists(csv_file_temp):
        try:
            existing_df = pd.read_csv(csv_file_temp, usecols=['problem_name'])
            completed_problems = set(existing_df['problem_name'].tolist())
            print(f"Found {len(completed_problems)} already completed problems in temp file. Resuming...")
        except Exception as e:
            print(f"Could not read existing temp CSV for resumption: {e}")

    # 4. Processing loop
    for name in problem_names:
        if name in completed_problems:
            continue
            
        if name in para_problem_names:
            index = para_problem_names.index(name)
            args = problem_argins[index] if index < len(problem_argins) else []
        else:
            args = None
            
        # Write current problem name before processing it to detect crashes
        with open(current_prob_file, 'w') as f:
            f.write(name)

        # Run each problem in a subprocess so that segfault/OOM during load only kills the
        # child; we exclude the problem and continue. No need to restart the whole script.
        args_json = json.dumps(args) if args is not None else "null"
        cmd = [sys.executable, os.path.abspath(__file__), "--single", name, args_json]
        try:
            ret = subprocess.run(cmd, cwd=cwd, timeout=None)
        except subprocess.TimeoutExpired:
            ret = subprocess.CompletedProcess(cmd, returncode=-1)

        if ret.returncode != 0:
            print(f"Problem {name} crashed or failed (exit {ret.returncode}). Excluding and continuing.")
            append_to_txt(exclude_file, name)
            if name not in problem_exclude:
                problem_exclude.append(name)
            if os.path.exists(current_prob_file):
                os.remove(current_prob_file)
            if os.path.exists(result_single_path):
                try:
                    os.remove(result_single_path)
                except Exception:
                    pass
            sys.stdout.flush()
            sys.stderr.flush()
            continue

        with open(result_single_path, "r") as f:
            info = json.load(f)
        try:
            os.remove(result_single_path)
        except Exception:
            pass

        def has_unknown_values(info_dict):
            for value in info_dict.values():
                if str(value).strip().lower() == 'unknown':
                    return True
            return False

        if not has_unknown_values(info):
            df_single = pd.DataFrame([info])
            if not os.path.exists(csv_file_temp):
                df_single.to_csv(csv_file_temp, index=False, na_rep='nan')
            else:
                df_single.to_csv(csv_file_temp, mode='a', header=False, index=False, na_rep='nan')
        else:
            print(f"Filtered out problem {name} due to 'unknown' values.")

        if os.path.exists(current_prob_file):
            os.remove(current_prob_file)

        sys.stdout.flush()
        sys.stderr.flush()

    # 5. Finalize: If we reached here, all problems are processed successfully.
    print("All problems processed. Finalizing output files...")
    if os.path.exists(csv_file_temp):
        shutil.move(csv_file_temp, csv_file)
    
    # Handle txt files renaming
    for base_name in ['feasibility_python', 'timeout_problems_python', 'log_python']:
        temp_path = os.path.join(saving_path, base_name + '_temp.txt')
        final_path = os.path.join(saving_path, base_name + '.txt')
        if os.path.exists(temp_path):
            shutil.move(temp_path, final_path)

    print("Script completed successfully.")

    # Close the log file
    log_file.close()

    sys.stdout = sys.__stdout__  # Reset stdout to default
    sys.stderr = sys.__stderr__  # Reset stderr to default