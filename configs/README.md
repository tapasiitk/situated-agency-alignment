# Config Layout

The root `configs/*.yaml` files are kept for backwards-compatible commands and
older docs. The subfolders are the cleaner study views:

- `base/`: generic or shared configs.
- `m0_ecology_calibration/`: Stage 0 / M0 ecology scout configs.
- `m1_frozen_mechanism/`: the frozen ecology selected for M1.
- `m1_legacy_scarcity/`: older scarcity-sweep M1 configs.
- `m2_intervention/`: KARMA and broken-mirror configs layered on the M1 frozen
  ecology.
- `smoke_debug/`: smoke, diagnostic, and historical debug configs.

For exact M1 reproduction, the compatibility path remains:

```bash
python train_karma.py \
  --config configs/m1_env_A_frozen_n6_ad030_rg075_zt25.yaml \
  --mode baseline \
  --seed 42
```
