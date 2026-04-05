# S2MPJ Python Subset

This repository provides a specialized Python-only subset of the [S2MPJ](https://github.com/GrattonToint/S2MPJ) collection.

## Contents

This repository preserves only the files relevant to Python users from the original source. These files are located in the `src/` directory:

- **`src/python_problems/`**: Directory containing the optimization problems converted to Python.
- **`src/list_of_python_problems`**: A listing of all available problems.
- **`src/s2mpjlib.py`**: Supporting library script.

## Configuration

The file `config.txt` in this directory controls how `s2mpj_select` filters problems (e.g., `variable_size` and `test_feasibility_problems`). See the comments in `config.txt` for a full description of each option.

When used through **OptiProfiler**, you can override these options at runtime without editing `config.txt`:

```python
from optiprofiler import set_plib_config, get_plib_config

# View the current effective configuration
print(get_plib_config('s2mpj'))

# Override at runtime (persists for the current Python process)
set_plib_config('s2mpj', variable_size='all', test_feasibility_problems=2)
```

You can also set the environment variables `S2MPJ_VARIABLE_SIZE` and `S2MPJ_TEST_FEASIBILITY_PROBLEMS` directly. Environment variables take precedence over `config.txt`.

## Maintenance

This repository is **automatically synchronized** with the upstream `GrattonToint/S2MPJ` repository via GitHub Actions. It checks for updates daily to ensure the problem set remains current.

For the full collection or other languages, please visit the [original repository](https://github.com/GrattonToint/S2MPJ).
