import mesa
import pandas as pd
import numpy as np
from smuggle_and_seek.model import SmuggleAndSeekGame

params = {
    "width": 2,
    "height": 2,
    "tom_customs": range(0,2,1),
    "tom_smuggler": range(0,2,1),
    "learning_speed": np.arange(0.1,1,0.1)
}

results = mesa.batch_run(
    SmuggleAndSeekGame,
    parameters=params,
    iterations=10,
    max_steps=1000,
    display_progress=True,
)

results_df = pd.DataFrame(results)
print(results_df)

for i in np.arange(0.1,1,0.1):
    customspoints_tom0vs0 = results_df[(results_df.learning_speed == i) & (results_df.tom_customs == 0) & (results_df.tom_smuggler == 0)]["customs points"]
    smugglerspoints_tom0vs0 = results_df[(results_df.learning_speed == i) &(results_df.tom_customs == 0) & (results_df.tom_smuggler == 0)]["smuggler points"]
    print(f"customs points on average tom0 vs tom0 ({i}): {sum(customspoints_tom0vs0)/len(customspoints_tom0vs0)}")
    print(f"smugglers points on average tom0 vs tom0 ({i}): {sum(smugglerspoints_tom0vs0)/len(smugglerspoints_tom0vs0)}")

    customspoints_tom1vs0 = results_df[(results_df.learning_speed == i) &(results_df.tom_customs == 1) & (results_df.tom_smuggler == 0)]["customs points"]
    smugglerspoints_tom0vs1 = results_df[(results_df.learning_speed == i) &(results_df.tom_customs == 1) & (results_df.tom_smuggler == 0)]["smuggler points"]
    print(f"customs points on average tom1 vs tom0 ({i}): {sum(customspoints_tom1vs0)/len(customspoints_tom1vs0)}")
    print(f"smugglers points on average tom1 vs tom0 ({i}): {sum(smugglerspoints_tom0vs1)/len(smugglerspoints_tom0vs1)}")




