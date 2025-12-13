# from gymnasium.envchecker import check_env
from gymnasium.utils.env_checker import check_env
# from pettingzoo.utils.env_checker import check_parallel_env
from .harvest_parallel import HarvestParallelEnv

# Register with Gymnasium/PettingZoo
try:
    import gymnasium.register as register
    register.register(
        id="KarmicHarvest-v0",
        entry_point="karmic_rl.envs:HarvestParallelEnv",
        max_episode_steps=1000,
    )
except:
    print("Gymnasium registration skipped (not available)")

print("KarmicHarvest environment registered!")
