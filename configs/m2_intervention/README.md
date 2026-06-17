# M2 Intervention Configs

M2 should differ from M1 only by the representational intervention layer. The
environment block below is copied from the M1 frozen ecology:

`configs/m1_env_A_frozen_n6_ad030_rg075_zt25.yaml`

Run the intervention configs with the matching CLI mode:

```bash
python train_karma.py \
  --config configs/m2_intervention/m2_env_A_frozen_n6_ad030_rg075_zt25_karma.yaml \
  --mode karma \
  --seed 42

python train_karma.py \
  --config configs/m2_intervention/m2_env_A_frozen_n6_ad030_rg075_zt25_broken.yaml \
  --mode broken \
  --seed 42
```
