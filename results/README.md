# Result Layout

Results are grouped by study stage:

- `m0_ecology_calibration/`: ecology scout CSV/JSON summaries.
- `m1_frozen_mechanism/`: current frozen-ecology M1 outputs.
- `m1_legacy_scarcity/`: older scarcity-sweep M1 outputs.
- `m2_intervention/`: KARMA and broken-mirror intervention outputs.
- `smoke_debug/`: smoke-test outputs.
- `synced_external/`: copied or synced artifacts from other machines.

Several older paths remain as compatibility symlinks. For example,
`results/m1_env_A_sc030` points into `results/m1_legacy_scarcity/`.

Checkpoint and rollout files can be large and remain ignored by `.gitignore`;
aggregate CSVs, summary JSONs, and final figures are the lightweight artifacts
to keep.
