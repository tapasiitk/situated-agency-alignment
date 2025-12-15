# from gymnasium.utils.env_checker import check_env
# from .harvest_parallel import HarvestParallelEnv
# from gymnasium.envs.registration import register

# # Register with Gymnasium
# try:
#     register(
#         id="KarmicHarvest-v0",
#         entry_point="karmic_rl.envs.harvest_parallel:HarvestParallelEnv",
#         max_episode_steps=1000,
#     )
#     print("KarmicHarvest environment registered!")
# except Exception as e:
#     print(f"Gymnasium registration failed: {e}")

from .harvest_dual import HarvestDualEnv

# No need for gymnasium.register() for PettingZoo Custom Envs
# We import the class directly in train_karma.py
