# OptiProfiler 2.0 compatibility

This line is the reviewed S2MPJ adapter for OptiProfiler 2.0.0. Provider API
version remains **1**. Upstream problem definitions, selection metadata and the
manual review of upstream updates are unchanged.

The adapter supplies selection and public Problem callbacks. Ordered feature
composition, counters and histories, scores, save/load, and eval_report/2 are
owned by core, not duplicated in adapters. No core-private class is accessed.
Python loaders return the exact public Problem class, not a subclass.

No trusted reference has been promoted for these wrapped instances, so
Problem.reference remains None and profiles keep the empirical default target.
Initial values, history minima, upstream solution vectors and metadata fbest
are not converted into references. A future fact needs raw `fun` and `maxcv`,
one of the two kinds `feasible` or `merit`, and a verifiable source under the
core reference contract.

Native modules load under an identity owned by this installation, so another
S2MPJ checkout in the same process cannot answer for this one, in either
import order. `sys.path` and the global names `python_problems` and `s2mpjlib`
are left alone. Native metadata fails closed: an undefined or impossible bound
never drops a constraint, and an unreadable linear constraint fails the load.

S2MPJ remains bundled with core, without a separate PyPI distribution.
Keep LICENCE.txt and THIRD_PARTY_NOTICES.md in the consuming wheel/sdist.
