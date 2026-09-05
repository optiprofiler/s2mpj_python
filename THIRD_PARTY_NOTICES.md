# S2MPJ sources and redistribution notice

## Upstream S2MPJ

The files in `src/` originate from [S2MPJ](https://github.com/GrattonToint/S2MPJ)
by Serge Gratton and Philippe L. Toint. The problem files retain their
individual mathematical-source references, SIF-input credits and translation
notices. These references must be preserved; this mirror does not claim
authorship of the original problem collection.

S2MPJ is licensed under **BSD-3-Clause**. The complete upstream
[`LICENCE.txt`](LICENCE.txt) is reproduced byte for byte, including:

> Copyright (c) 2026, S. Gratton and Ph. L. Toint

Redistributors must retain that copyright notice, the three conditions and
the disclaimer. The authors' names may not be used to imply endorsement.
The license text, rather than this explanatory notice, governs upstream use.

- Upstream license revision: `fea6a70048eaad28b13a08703ddbfdbf65cd9c30`.
- [Exact license source](https://github.com/GrattonToint/S2MPJ/blob/fea6a70048eaad28b13a08703ddbfdbf65cd9c30/LICENCE.txt).
- SHA-256: `a8636fc42ac474fc85fbf451c6a0316f6cbd9efa9031d549797dec6b43e9e5b4`.

## Reviewed source snapshot

The license was adopted separately from numerical source updates. On
2026-09-05 the tracked source subset was compared with upstream at the
license revision above:

- Provider baseline: `9ebd196908a0085b3f07ef3c86fd34c88b1721fa`.
- Unchanged Git tree for `src/`: `531090480933a017cf7a65010eea6bb0ebf62ffb`.
- 1,105 of 1,106 tracked source files match upstream byte for byte.
- `src/s2mpjlib.py` is an upstream-derived supporting library with a retained
  local version: two derivative-product call sites use `feval(...)` where
  current upstream uses calls to methods of `self`. This historical
  difference is recorded for a separate numerical review, not promoted here.

The license revision is **not** a claim that every file was synchronized to
that revision. The provider Git commit freezes its adapter, metadata and
retained source together. The daily upstream workflow reports candidates,
including license differences; adopting changes always requires review.

## OptiProfiler additions

`__init__.py`, `s2mpj_tools.py`, configuration, generated
`probinfo_python.csv`, tests and maintenance workflows provide the
OptiProfiler integration. Generated metadata is derived from the
wrapped upstream problems; it is not a separately authored problem collection.

Git history records adapter commits by `Huang_Mac`, but commit authorship
alone does not establish the copyright ownership of every contribution.
The upstream license notice does not assign those independent additions to
Gratton and Toint, or automatically license them. This update does not assign
new ownership or choose a repository-wide license for independent additions;
maintainers must confirm that separately before claiming a license for the
whole combined distribution.

## Citation and distribution

Please cite S. Gratton and Ph. L. Toint, *S2MPJ and CUTEst optimization
problems for Matlab, Python and Julia*, Optimization Methods and Software
40(4), 871-903 (2025), [doi:10.1080/10556788.2025.2490640](https://doi.org/10.1080/10556788.2025.2490640).
The [authors' preprint](https://arxiv.org/abs/2407.07812) and
[upstream documentation](https://github.com/GrattonToint/S2MPJ/blob/fea6a70048eaad28b13a08703ddbfdbf65cd9c30/s2mpj.pdf)
describe the collection.

Carry `LICENCE.txt` and this notice alongside the bundled source in every
Python wheel/sdist and source archive.
This provider has no independent distribution metadata or version. It remains
bundled with OptiProfiler; its licensing files use extensions covered by the
core's existing source-package manifest.
