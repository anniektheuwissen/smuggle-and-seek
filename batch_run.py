import mesa
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from smuggle_and_seek.model import SmuggleAndSeekGame

params = {
    "k": 2,
    "l": 2,
    "m": 5,
    "tom_police": range(0,3,1),
    "tom_smuggler": range(0,2,1),
    "learning_speed1": 0.2,
    "learning_speed2": 0.1
}

results = mesa.batch_run(
    SmuggleAndSeekGame,
    parameters=params,
    iterations=100,
    display_progress=True,
)


# Create graph from csv:
results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv('results.csv')

results_0vs0 = results_df[(results_df["tom_police"] == 0) & (results_df["tom_smuggler"] == 0)]
results_1vs0 = results_df[(results_df["tom_police"] == 1) & (results_df["tom_smuggler"] == 0)]
results_0vs1 = results_df[(results_df["tom_police"] == 0) & (results_df["tom_smuggler"] == 1)]
results_1vs1 = results_df[(results_df["tom_police"] == 1) & (results_df["tom_smuggler"] == 1)]
results_2vs0 = results_df[(results_df["tom_police"] == 2) & (results_df["tom_smuggler"] == 0)]
results_2vs1 = results_df[(results_df["tom_police"] == 2) & (results_df["tom_smuggler"] == 1)]
print(results_0vs0)
print(results_1vs0)
print(results_0vs1)
print(results_1vs1)
print(results_2vs0)
print(results_2vs1)


for data in ["police points", "smuggler points", "successful checks", "successful smuggles", "caught packages", "smuggled packages", "features used by smuggler that are not preferred"]:
    print(data)
    t_stat_1vs0, p_val_1vs0 = stats.ttest_ind(results_0vs0[data], results_1vs0[data])
    print(f"0vs0 against 1vs0: {t_stat_1vs0}, {p_val_1vs0}")

    t_stat_0vs1, p_val_0vs1 = stats.ttest_ind(results_0vs0[data], results_0vs1[data])
    print(f"0vs0 against 0vs1: {t_stat_0vs1}, {p_val_0vs1}")

    t_stat_2vs0, p_val_2vs0 = stats.ttest_ind(results_0vs0[data], results_2vs0[data])
    print(f"0vs0 against 2vs0: {t_stat_2vs0}, {p_val_2vs0}")

    t_stat_2vs1, p_val_2vs1 = stats.ttest_ind(results_1vs1[data], results_2vs1[data])
    print(f"1vs1 against 2vs1: {t_stat_2vs1}, {p_val_2vs1}")
    
    fig, ax = plt.subplots(figsize=(16,8))
    boxplot = ax.boxplot([results_0vs0[data], results_1vs0[data], results_0vs1[data], results_2vs0[data], results_1vs1[data], results_2vs1[data]],
                            labels=['ToM0 police vs\n ToM0 smuggler', 'ToM1 police vs\n ToM0 smuggler', 'ToM0 police vs\n ToM1 smuggler', 'ToM2 police vs\n ToM0 smuggler', 'ToM1 police vs\n ToM1 smuggler', 'ToM2 police vs\n ToM1 smuggler'], patch_artist=True, medianprops={'color': 'black'}
                            )
    colors = ['#A8A9AD', '#728FCE', '#728FCE', '#728FCE', '#A8A9AD', '#728FCE']
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)
    plt.ylabel(data)
    plt.title(f"Number of {data} after 1000 days")

    plt.text(2.27, np.median(results_1vs0[data]-2), f'p-value: {"{:.1e}".format(p_val_1vs0)}')
    plt.text(2.18, np.median(results_0vs1[data]-2), f'p-value: {"{:.1e}".format(p_val_0vs1)}')
    plt.text(3.19, np.median(results_2vs0[data]-2), f'p-value: {"{:.1e}".format(p_val_2vs0)}')
    plt.text(5.19, np.median(results_2vs1[data]-2), f'p-value: {"{:.1e}".format(p_val_2vs1)}')

    plt.show()
