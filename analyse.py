import wandb
api = wandb.Api(79718dcc7810d54f4d12457ef36c5d3cd60276d7)
run = api.run("/ratht-iitk/karma-mirror-test/runs/81w46dhh")
print(run.history())