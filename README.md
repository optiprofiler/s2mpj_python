# S2MPJ Python Subset

This repository provides a specialized Python-only subset of the [S2MPJ](https://github.com/GrattonToint/S2MPJ) collection.

## Contents

This repository preserves only the files relevant to Python users from the original source. These files are located in the `src/` directory:

- **`src/python_problems/`**: Directory containing the optimization problems converted to Python.
- **`src/list_of_python_problems`**: A listing of all available problems.
- **`src/s2mpjlib.py`**: Supporting library script.

## OptiProfiler Lifecycle

S2MPJ is the bundled default Python problem library in OptiProfiler. Ordinary
users install it with the core `optiprofiler` distribution and should not
install this repository as a separate Python package. It is discoverable under
the public name `s2mpj`:

```python
from optiprofiler import list_problem_libraries

assert "s2mpj" in list_problem_libraries()
```

Select it with `benchmark(..., plibs=["s2mpj"])`; no custom filesystem path is
needed. Removing `optiprofiler` also removes its bundled S2MPJ files, but it
does not remove benchmark output or other user data.

This repository keeps a reviewed S2MPJ snapshot for OptiProfiler maintenance.
An automated workflow checks upstream and reports differences, but it never
changes `src/`, metadata, or the OptiProfiler lock. Users receive a new S2MPJ
snapshot only after maintainers review the candidate, commit it explicitly,
update the locked gitlink, and publish or install a matching core revision.

## Configuration

The file `config.txt` in this directory controls how `s2mpj_select` filters problems (e.g., `variable_size` and `test_feasibility_problems`). See the comments in `config.txt` for a full description of each option.

This repository keeps the legacy `s2mpj_load` / `s2mpj_select` interface
while also exposing the same API-v1 adapter callbacks used by separately
installed problem-library plugins.

For a reproducible OptiProfiler experiment, pass the options explicitly for
this run:

```python
from optiprofiler import benchmark

benchmark(
    solvers,
    plibs=['s2mpj'],
    plib_options={
        's2mpj': {
            'variable_size': 'all',
            'test_feasibility_problems': 2,
        },
    },
)
```

OptiProfiler stores the validated effective mapping with the experiment. For a
process-level default shared by subsequent calls, the compatibility API remains
available:

```python
from optiprofiler import set_plib_config, get_plib_config

# View the current effective configuration
print(get_plib_config('s2mpj'))

# Override subsequent calls in the current Python process
set_plib_config('s2mpj', variable_size='all', test_feasibility_problems=2)
```

The precedence is per-run `plib_options`, process-level `set_plib_config`,
environment variables, `config.txt`, then built-in defaults. You can also set
`S2MPJ_VARIABLE_SIZE` and `S2MPJ_TEST_FEASIBILITY_PROBLEMS` directly. The
adapter merges these layers first and validates the final mapping once, so an
explicit valid per-run value can replace an invalid lower-priority value.

## Testing

The `CI` workflow runs daily and on pushes. It checks the OptiProfiler adapter layer by:

- selecting a small set of representative `u`, `b`, `l`, and `n` problems;
- loading each selected problem through `s2mpj_load`;
- evaluating `fun`, `cub`, and `ceq` at the initial point;
- checking `variable_size` and `test_feasibility_problems` environment overrides;
- checking the OptiProfiler API-v1 adapter callbacks used by the core loader;
- checking nonlinear lower/upper/two-sided constraint Hessians against Jacobian
  differences, including stacking and multiplier order;
- sampling a few additional small problems each day with at most two numerical-library threads.

Locally, from this repository:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Maintenance

`Check S2MPJ Upstream` compares the managed Python subset with the latest
`GrattonToint/S2MPJ` revision every day. A difference creates or updates an
`upstream-update` issue and uploads a report. The workflow has no permission to
push source changes. `Collect Info` is manual and uploads candidate metadata as
an artifact; adopting either source or metadata requires a reviewed commit.

## Provenance and Citation

The files under `src/` originate from [S2MPJ](https://github.com/GrattonToint/S2MPJ)
by Serge Gratton and Philippe L. Toint. Their **BSD-3-Clause** license is
preserved verbatim in [LICENCE.txt](LICENCE.txt). The
[third-party notice](THIRD_PARTY_NOTICES.md) records the exact license revision,
source comparison and attribution, and distinguishes the independently added
OptiProfiler adapter and metadata. The upstream copyright notice does not
assign ownership of those independent additions.

Please cite S. Gratton and Ph. L. Toint, *S2MPJ and CUTEst optimization
problems for Matlab, Python and Julia*, Optimization Methods and Software
40(4), 871-903 (2025), [doi:10.1080/10556788.2025.2490640](https://doi.org/10.1080/10556788.2025.2490640).

Distributions must carry `LICENCE.txt` and `THIRD_PARTY_NOTICES.md` together
with the source. CI checks their exact contents in source archives and in
the core wheel/sdist built with this bundled provider. To check a built archive:

```bash
python3 tests/check_distribution.py /path/to/archive.whl
```

For the full collection or other languages, please visit the [original repository](https://github.com/GrattonToint/S2MPJ).
