# Validation report for this generated repository

Validation performed before packaging:

- `PYTHONPATH=src pytest -q` -> **16 passed**;
- `python -m compileall -q src scripts` -> passed;
- editable installation with `pip install -e . --no-build-isolation` -> passed in the build environment;
- installed CLI `tcstf-eval` -> passed;
- installed CLI `tcstf-train --help` -> passed;
- `scripts/audit_paper_reference.py` verified:
  - M5 nGap95 reduction vs Gen-DFL = 37.3%;
  - battery nGap95 reduction vs Gen-DFL = 40.3%;
  - worst nGap reduction = 38.8%;
  - default `d=408, r=8, M=500` raw payloads = 796.875 KiB and 15.625 KiB;
  - payload reduction = 98.0392% (reported as 98.0%).

The tests validate code invariants and arithmetic consistency. They do **not** claim to reproduce the paper's locked numerical experiment archive, which was not contained in the supplied manuscript files.
