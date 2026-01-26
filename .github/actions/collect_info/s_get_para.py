import os
import re
import shutil
from pathlib import Path

def s_get_para(language='matlab', max_para=500):
    """
    Extracts parametric problems from MATLAB files or other specified language files.
    """

    # Get the current working directory
    cwd = os.getcwd()

    # Search and copy the file 'list_of_matlab_problems' to the path './list_of_parametric_problems.txt'
    src_file = 'optiprofiler/problems/s2mpj/src/list_of_' + language + '_problems'
    dst_file = 'list_of_parametric_problems_' + language + '.txt'
    shutil.copy(src_file, dst_file)

    # Path to the text file containing the list of names
    names_file_path = os.path.join(cwd, dst_file)

    # Directory where the language files are located
    language_files_folder = os.path.join(cwd, 'optiprofiler/problems/s2mpj/src', language + '_problems')

    # Output file
    output_file_path = os.path.join(cwd, 'list_of_parametric_problems_with_parameters_' + language + '.txt')

    # First Scan the language_files_folder, if the m file contains string '$-PARAMETER', then add the name before .m (or .py) to the txt file names_file_path. Finally sort the names in the names_file_path.
    # Open the names file for writing
    if language == 'matlab':
        suffix = '.m'
    elif language == 'python':
        suffix = '.py'
    else:
        raise ValueError("Unsupported language. Supported languages are 'matlab' and 'python'.")

    with open(names_file_path, 'w') as output_file:
        # Process each name
        for file in os.listdir(language_files_folder):
            if file.endswith(suffix):
                # Construct the full path to the corresponding .m (or .py) file
                language_file_path = os.path.join(language_files_folder, file)

                # Check if the .m (or .py) file exists
                if not os.path.isfile(language_file_path):
                    print(f"Warning: File not found for '{file}': {language_file_path}")
                    continue  # Skip to the next name if the file doesn't exist

                # Read the content of the .m (or .py) file
                with open(language_file_path, 'r') as m_file:
                    content = m_file.readlines()

                # Process each line to find the variable assignment and parameters
                for line in content:
                    # If '$-PARAMETER' is in the line, write the name to the output file
                    if '$-PARAMETER' in line:
                        if language == 'matlab':
                            name = file[:-2]  # Remove the '.m' (or .py) extension
                        elif language == 'python':
                            name = file[:-3]
                        output_file.write(f'{name}\n')
                        print(f"Found '$-PARAMETER' in '{file}'. Added '{name}' to the list.")
                        break  # Break after finding the first '$-PARAMETER' line

    # Sort the names in the names file
    with open(names_file_path, 'r') as f:
        names = sorted([line.strip() for line in f if line.strip()])
    with open(names_file_path, 'w') as f:
        f.write('\n'.join(names))

    print(f"First scan complete. Results saved to '{names_file_path}'.")

    # Read the list of names from the names file
    with open(names_file_path, 'r') as f:
        names = [line.strip() for line in f if line.strip()]

    # Open the output file for writing
    with open(output_file_path, 'w') as output_file:
        # Process each name
        for name in names:
            numbers_set = set()
            variable_name = None  # Variable to store the VariableName assigned to varargin{1}

            # Consider some special cases that we already know
            if name == 'NUFFIELD':
                output_file.write(f'NUFFIELD,{{5.0}}{{10,20,30,40,100}}\n')
                continue
            elif name == 'TRAINF':
                output_file.write(f'TRAINF,{{1.5}}{{2}}{{11,51,101,201,501}}\n')
                continue
            elif name == 'QPBAND':
                continue
            elif name == 'WACHBIEG':
                continue

            # Construct the full path to the corresponding .m (or .py) file
            language_file_path = os.path.join(language_files_folder, f'{name}' + suffix)

            # Check if the .m (.py) file exists
            if not os.path.isfile(language_file_path):
                print(f"Warning: File not found for '{name}': {language_file_path}")
                continue  # Skip to the next name if the file doesn't exist

            # Read the content of the .m (.py) file
            with open(language_file_path, 'r') as m_file:
                content = m_file.readlines()

            # Process each line to find the variable assignment and parameters
            for line in content:
                # Remove comments and leading/trailing whitespace
                if language == 'matlab':
                    code_line = line.split('%', 1)[0].strip()
                    pattern = r"v_\(\s*['\"]([^'\"]+)['\"]\s*\)\s*=\s*varargin\{\s*1\s*\};?"
                    arg_sign = 'varargin{1}'
                elif language == 'python':
                    code_line = line.split('#', 1)[0].strip()
                    pattern = r"v_\[\s*['\"]([^'\"]+)['\"]\s*\]\s*=\s*(?:int|float|str)?\s*\(?\s*args\[0\]\s*\)?"
                    arg_sign = 'args[0]'

                # Check if arg_sign is in the line
                if arg_sign in code_line:
                    # Use regex to find v_('VariableName') = varargin{1};
                    match = re.search(pattern, code_line)
                    if match:
                        variable_name = match.group(1)
                        # Since variable_name may include special characters, escape them for regex
                        variable_name_regex = re.escape(variable_name)
                        break  # Break after finding the first variable assignment

            # If variable_name was found, search for $-PARAMETER lines associated with it
            if variable_name:
                for line in content:
                    if '$-PARAMETER' in line:
                        param_line = line.strip()
                        # Updated regex pattern
                        if language == 'matlab':
                            pattern = rf"%\s*\w+\s+({variable_name_regex}(?:[+\-*/]\w+)*)\s+([-+]?\d*\.?\d+)\s*\$-PARAMETER"
                        elif language == 'python':
                            pattern = rf"#\s*\w+\s+({variable_name_regex}(?:[+\-*/]\w+)*)\s+([-+]?\d*\.?\d+)\s*\$-PARAMETER"
                        param_match = re.match(pattern, param_line)
                        if param_match:
                            var_in_line = param_match.group(1)
                            number = param_match.group(2)
                            # If the float(number) is a number smaller than max_para, and not the case like 5.0 (it should be 5), add it to the set
                            if float(number) <= max_para:
                                if '.' not in number:
                                    numbers_set.add(number)
                                else:
                                    # Add the number and a 'Check' to the problem_name
                                    numbers_set.add(number)
                                    name += 'Check'


            # If numbers were found, sort them and write to the output file
            if numbers_set:
                # Convert numbers to floats for sorting
                numbers_list = sorted(numbers_set, key=float)
                # Join the numbers with commas
                numbers_str = ','.join(numbers_list)
                # Write the line to the output file
                output_file.write(f'{name},{numbers_str}\n')

    # Delete the file 'list_of_parametric_problems.txt'
    os.remove(names_file_path)
    print(f"Second scan complete. Deleted '{names_file_path}'.")

    print(f"Extraction complete. Results saved to '{output_file_path}'.")
    print("You need to check all the problems with 'Check' in the name.")


if __name__ == "__main__":
    import sys
    language_arg = 'matlab'  # 默认值
    max_para_arg = 500
    if len(sys.argv) > 1:
        language_arg = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            max_para_arg = int(sys.argv[2])
        except ValueError:
            print("Invalid max_para value. Using default 500.")
    s_get_para(language_arg, max_para_arg)