# H2 ANI Active Learning (One-command PBS)

This folder now provides an ADL-style one-command entrypoint for **ANI-only**, direct-learning active learning on H2.

- Published method paper (in this workspace): `physics-informed-active-learning-for-accelerating-quantum-chemical-simulations.pdf`
- DOI: `10.1021/acs.jctc.4c00821`

## Core Layout

- `configs/h2_ani_al.yaml`: single source of truth for molecule, reference method, AL params, PBS params.
- `scripts/run_h2_al.py`: stable entrypoint (delegates to the new runner).
- `scripts/h2_al_runner.py`: new implementation (`local` execution + `pbs` submit/wait logic).
- `scripts/inspect_al_results.py`: stable inspector entrypoint (al_info-first).
- `scripts/inspect_al_results_core.py`: new result parser and acceptance checker.
- `inputs/h2.xyz`, `inputs/h2_freq.json`: default H2 inputs.

## One-command Submit

Smoke submit (default no-wait):

```bash
bash scripts/run_smoke.sh
```

Full submit (default no-wait):

```bash
bash scripts/run_full.sh
```

Equivalent direct command:

```bash
python3 scripts/run_h2_al.py \
  --config configs/h2_ani_al.yaml \
  --mode smoke \
  --submit-mode pbs \
  --no-wait
```

## Check Status and Results

After submission, the run directory contains:

- `submission.json`: includes `job_id`, job script path, stdout/stderr paths.
- `status.json`: written by the in-job local execution branch.
- `run_config.json`: fully resolved runtime config snapshot.
- `al_info.json`: native MLatom AL output.

Inspect acceptance:

```bash
python3 scripts/inspect_al_results.py --run-dir runs/h2_ani_smoke --min-new-points 5
```

Use `runs/h2_ani_full` for full mode.

## Config Notes (`configs/h2_ani_al.yaml`)

- `system.xyz` / `system.freq_json`: molecule inputs (replace these when changing system from H2).
- `reference.refmethod` / `reference.qmprog`: reference labels (default `B3LYP/6-31G*` + `gaussian`).
- `training.ml_model_type`: fixed to `ANI` in this workflow.
- `active_learning.*`: tutorial-style AL knobs (`init_train_set_size`, `max_sampled_points`, `min_new_points`, `max_excess`, `min_excess`).
- `cluster.*`: PBS submission settings (`queue`, `nodes`, `ppn`, `walltime`, `submit_command`, `env_blocks`).

## Optional Local Dry-run

Validate config parsing and output paths without running AL:

```bash
python3 scripts/run_h2_al.py \
  --config configs/h2_ani_al.yaml \
  --mode smoke \
  --submit-mode local \
  --dry-run
```
