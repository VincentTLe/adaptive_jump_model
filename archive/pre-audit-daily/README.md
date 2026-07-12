# Frozen Pre-Audit Daily Experiments

Status: frozen and unsupported. These scripts preserve the daily experiments
that existed before the repository-wide protocol audit. They are not part of
the active research pipeline and are not guaranteed to run from this location.

- `p0_daily_baseline.py` is an approximate Yahoo-data experiment. It does not
  reproduce the paper's 3000-day daily online decoder, monthly eight-year
  validation chain, HMM protocol, or local risk-free returns.
- `p1_adaptive_lambda.py` is an exploratory scalar-penalty experiment whose
  reported variants did not establish a robust improvement over fixed JM.
- `p2_asymmetric.py` is withdrawn. With zero diagonal costs, its two-state
  asymmetric transition objective reduces to symmetric switching plus an
  endpoint term; its duration-based parameterization is also not valid for the
  fitted-loss scale.

Do not cite outputs from these files as replication or confirmatory evidence.
Generated CSV, PNG, and HTML results are excluded from the archive and remain
available through Git history or the local pre-cleanup snapshot. The shared
time-varying DP stays in the active package because it will be validated and
reused by the replacement protocol.
