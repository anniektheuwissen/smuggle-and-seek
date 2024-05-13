import mesa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from smuggle_and_seek.model import SmuggleAndSeekGame

params = {
    "width": 2,
    "height": 2,
    "tom_customs": range(0,2,1),
    "tom_smuggler": range(0,2,1),
    "learning_speed": 0.2
}

results = mesa.batch_run(
    SmuggleAndSeekGame,
    parameters=params,
    iterations=100,
    display_progress=True,
)

results_df = pd.DataFrame(results)
print(results_df)

results_0vs0 = results_df[(results_df["tom_customs"] == 0) & (results_df["tom_smuggler"] == 0)]
results_1vs0 = results_df[(results_df["tom_customs"] == 1) & (results_df["tom_smuggler"] == 0)]
results_0vs1 = results_df[(results_df["tom_customs"] == 0) & (results_df["tom_smuggler"] == 1)]
print(results_0vs0)
print(results_1vs0)
print(results_0vs1)

for data in ["customs points","smuggler points", "successful checks", "successful smuggles", "caught packages", "smuggled packages", "nonpreferences used"]:
    fig, ax = plt.subplots()
    plt.hist(results_0vs0[data], bins=20, alpha=0.5, label='0vs0')
    plt.hist(results_1vs0[data], bins=20, alpha=0.5, label='1vs0')
    plt.hist(results_0vs1[data], bins=20, alpha=0.5, label='0vs1')
    plt.legend()
    plt.show()

# fig, ax = plt.subplots()
# plt.violinplot([results_0vs0["customs points"], results_1vs0["customs points"], results_0vs1["customs points"]])
# ax.set_xticklabels(['0vs0', '1vs0', '0vs1'])
# plt.show()




# results_0vs0 = []
# results_1vs0 = []
# results_0vs1 = []

# for data in ["customs points","smuggler points", "successful checks", "successful smuggles", "caught packages", "smuggled packages", "nonpreferences used"]:
#     array = results_df[(results_df.tom_customs == 0) & (results_df.tom_smuggler == 0)][data]
#     print(f"{data} tom0 vs tom0: {sum(array)/len(array)}")
#     results_0vs0.append(sum(array)/len(array))

# print("\n")

# for data in ["customs points","smuggler points", "successful checks", "successful smuggles", "caught packages", "smuggled packages", "nonpreferences used"]:
#     array = results_df[(results_df.tom_customs == 1) & (results_df.tom_smuggler == 0)][data]
#     print(f"{data} tom1 vs tom0: {sum(array)/len(array)}")
#     results_1vs0.append(sum(array)/len(array))

# print("\n")

# for data in ["customs points","smuggler points", "successful checks", "successful smuggles", "caught packages", "smuggled packages", "nonpreferences used"]:
#     array = results_df[(results_df.tom_customs == 0) & (results_df.tom_smuggler == 1)][data]
#     print(f"{data} tom0 vs tom1: {sum(array)/len(array)*100}%")
#     results_0vs1.append(sum(array)/len(array))

# print("\n Results customs0 vs smuggler0 --> customs1 vs smuggler0")

# for (idx, data) in enumerate(["customs points","smuggler points", "successful checks", "successful smuggles", "caught packages", "smuggled packages", "nonpreferences used"]):
#     difference = results_1vs0[idx] - results_0vs0[idx]
#     print(f"{data}: {difference/abs(results_0vs0[idx])*100}%")

# print("\n Results customs0 vs smuggler0 --> customs0 vs smuggler1")

# for (idx, data) in enumerate(["customs points","smuggler points", "successful checks", "successful smuggles", "caught packages", "smuggled packages", "nonpreferences used"]):
#     difference = results_0vs1[idx] - results_0vs0[idx]
#     print(f"{data}: {difference/abs(results_0vs0[idx])*100}%")
