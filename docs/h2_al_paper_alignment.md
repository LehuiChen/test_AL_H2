# Tutorial Workflow vs. Paper Methodology

This note maps the H2 tutorial reproduction to the active learning loop reported in:

- Hou et al., *J. Chem. Theory Comput.* 2024, 20, 7744-7754
- DOI: `10.1021/acs.jctc.4c00821`

## Loop Mapping

1. Sampling
- Tutorial implementation: Wigner initial condition sampling around equilibrium H2 (`inputs/h2_freq.json`).
- Paper concept: physics-informed trajectory or coordinate-space sampling to preferentially visit chemically relevant regions.

2. Uncertainty Quantification (UQ)
- Tutorial implementation: MLatom AL loop with `uq_threshold`; adaptive update controlled by `max_excess/min_excess`.
- Paper concept: query by uncertainty to identify candidate configurations for expensive reference labeling.

3. Trigger Labeling
- Tutorial implementation: points with uncertainty above threshold are labeled by reference method (`B3LYP/6-31G*` with Gaussian).
- Paper concept: selectively run high-level electronic structure only where model confidence is low.

4. Incremental Retraining
- Tutorial implementation: retraining is executed after new labels are added; history tracked in `al_iteration_history.json`.
- Paper concept: iterative model refinement with progressively expanded, informative datasets.

5. Convergence / Stop
- Tutorial implementation: stop when `new_points < min_new_points` (default 5) and monitor trend in new points.
- Paper concept: stop once most sampled configurations are confidently predicted and only few uncertain points remain.

## Practical Acceptance Signals

- `al_info.json` exists and records run metadata.
- `labeled_db.json` grows over iterations.
- Newly added points trend downward and eventually trigger the stop criterion.
