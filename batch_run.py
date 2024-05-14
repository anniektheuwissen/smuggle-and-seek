import mesa
import pandas as pd
import numpy as np
from scipy import stats
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
results_df.to_csv('results.csv')

results_0vs0 = results_df[(results_df["tom_customs"] == 0) & (results_df["tom_smuggler"] == 0)]
results_1vs0 = results_df[(results_df["tom_customs"] == 1) & (results_df["tom_smuggler"] == 0)]
results_0vs1 = results_df[(results_df["tom_customs"] == 0) & (results_df["tom_smuggler"] == 1)]
print(results_0vs0)
print(results_1vs0)
print(results_0vs1)

# for data in ["customs points","smuggler points", "successful checks", "successful smuggles", "caught packages", "smuggled packages", "nonpreferences used"]:
#     fig, ax = plt.subplots()
#     plt.hist(results_0vs0[data], bins=20, alpha=0.5, label='0vs0')
#     plt.hist(results_1vs0[data], bins=20, alpha=0.5, label='1vs0')
#     plt.hist(results_0vs1[data], bins=20, alpha=0.5, label='0vs1')
#     plt.legend()
#     plt.show()


for data in ["customs points", "smuggler points", "successful checks", "successful smuggles", "caught packages", "smuggled packages", "features used by smuggler that are not preferred"]:
    print(data)
    t_stat_1vs0, p_val_1vs0 = stats.ttest_ind(results_0vs0[data], results_1vs0[data])
    print(f"0vs0 against 1vs0: {t_stat_1vs0}, {p_val_1vs0}")

    t_stat_0vs1, p_val_0vs1 = stats.ttest_ind(results_0vs0[data], results_0vs1[data])
    print(f"0vs0 against 0vs1: {t_stat_0vs1}, {p_val_0vs1}")
    
    fig, ax = plt.subplots(figsize=(8,5))
    boxplot = ax.boxplot([results_0vs0[data], results_1vs0[data], results_0vs1[data]], 
                            labels=['ToM0 customs vs\n ToM0 smuggler', 'ToM1 customs vs\n ToM0 smuggler', 'ToM0 customs vs\n ToM1 smuggler'], patch_artist=True, medianprops={'color': 'black'}
                            )
    colors = ['#A8A9AD', '#728FCE', '#728FCE']
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)
    plt.ylabel(data)
    plt.title(f"Number of {data} after 1000 days")

    plt.text(2.2, np.median(results_1vs0[data]-2), f'p-value: {"{:.1e}".format(p_val_1vs0)}')
    plt.text(2.25, np.median(results_0vs1[data]-2), f'p-value: {"{:.1e}".format(p_val_0vs1)}')

    plt.show()




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
