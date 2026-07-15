from .s2mpj_tools import (
    s2mpj_collect_info,
    s2mpj_get_default_options,
    s2mpj_load,
    s2mpj_select,
    s2mpj_validate_options,
)


def _plugin_select(problem_options, library_options):
    return s2mpj_select(problem_options, library_options=library_options)


def _plugin_load(problem_name, library_options):
    return s2mpj_load(problem_name, library_options=library_options)


def get_problem_library():
    """Return the OptiProfiler problem-library plugin for S2MPJ.

    S2MPJ remains bundled with OptiProfiler in this stage. The factory is kept
    here so the adapter can be validated against the same API v1 contract used
    by independently installed libraries.
    """

    from optiprofiler import ProblemLibraryPlugin

    return ProblemLibraryPlugin(
        name='s2mpj',
        api_version=1,
        select=_plugin_select,
        load=_plugin_load,
        collect_info=s2mpj_collect_info,
        get_default_options=s2mpj_get_default_options,
        validate_options=s2mpj_validate_options,
    )


__all__ = [
    'get_problem_library',
    's2mpj_collect_info',
    's2mpj_get_default_options',
    's2mpj_load',
    's2mpj_select',
    's2mpj_validate_options',
]
