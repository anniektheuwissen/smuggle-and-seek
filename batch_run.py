import mesa
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from smuggle_and_seek.model import SmuggleAndSeekGame

def barplot_annotate_brackets(num1, num2, data, center, height, dh=.05, barh=.025):
    """ 
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    ax_y0, ax_y1 = plt.gca().get_ylim()
    barh *= (ax_y1 - ax_y0)

    for i in range(len(data)):
        if type(data[i]) is str:
            text = data
        else:
            # * is p < 0.05
            # ** is p < 0.005
            # *** is p < 0.0005
            # etc.
            text = ''
            p = .05

            while data[i] < p:
                text += '*'
                p /= 10.

                if len(text) == 3:
                    break

            if len(text) == 0:
                text = 'n. s.'

        lx, ly = center[num1[i]], height[num1[i]]
        rx, ry = center[num2[i]], height[num2[i]]

        dh[i] *= (ax_y1 - ax_y0)

        y = max(ly, ry) + dh[i]

        barx = [lx, lx, rx, rx]
        bary = [y, y+barh, y+barh, y]
        mid = ((lx+rx)/2, y+barh)

        plt.plot(barx, bary, c='black')

        kwargs = dict(ha='center', va='bottom')

        plt.text(*mid, text, **kwargs)


params = {
    "k": 2,
    "l": 2,
    "m": 5,
    "tom_police": range(0,3,1),
    "tom_smuggler": range(0,2,1),
    "learning_speed1": 0.2,
    "learning_speed2": 0.05
}

results = mesa.batch_run(
    SmuggleAndSeekGame,
    parameters=params,
    iterations=10,
    display_progress=True,
)


# Create graph from csv:
results_df = pd.DataFrame(results)
results_df.to_csv('results.csv')

results_0vs0 = results_df[(results_df["tom_police"] == 0) & (results_df["tom_smuggler"] == 0)]
results_1vs0 = results_df[(results_df["tom_police"] == 1) & (results_df["tom_smuggler"] == 0)]
results_0vs1 = results_df[(results_df["tom_police"] == 0) & (results_df["tom_smuggler"] == 1)]
results_1vs1 = results_df[(results_df["tom_police"] == 1) & (results_df["tom_smuggler"] == 1)]
results_2vs0 = results_df[(results_df["tom_police"] == 2) & (results_df["tom_smuggler"] == 0)]
results_2vs1 = results_df[(results_df["tom_police"] == 2) & (results_df["tom_smuggler"] == 1)]


for data in ["police points", "smuggler points", "successful checks", "successful smuggles", "caught packages", "smuggled packages", "features used by smuggler that are not preferred"]:
    
    t_stat_1vs0, p_val_1vs0 = stats.ttest_ind(results_0vs0[data], results_1vs0[data])
    t_stat_0vs1, p_val_0vs1 = stats.ttest_ind(results_0vs0[data], results_0vs1[data])
    t_stat_2vs0, p_val_2vs0 = stats.ttest_ind(results_0vs0[data], results_2vs0[data])
    t_stat_2vs1, p_val_2vs1 = stats.ttest_ind(results_1vs1[data], results_2vs1[data])
    
    fig, ax = plt.subplots(figsize=(16,8))
    boxplot = ax.boxplot([results_0vs0[data], results_1vs0[data], results_0vs1[data], results_2vs0[data], results_1vs1[data], results_2vs1[data]],
                            labels=['ToM0 police vs\n ToM0 smuggler', 'ToM1 police vs\n ToM0 smuggler', 'ToM0 police vs\n ToM1 smuggler', 'ToM2 police vs\n ToM0 smuggler', 'ToM1 police vs\n ToM1 smuggler', 'ToM2 police vs\n ToM1 smuggler'], patch_artist=True, medianprops={'color': 'black'}
                            )
    colors = ['#A8A9AD', '#728FCE', '#728FCE', '#728FCE', '#A8A9AD', '#728FCE']
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)
    plt.ylabel(data)
    plt.title(f"Number of {data} after 1000 days")

    # plt.text(2.27, np.median(results_1vs0[data]-2), f'p-value: {"{:.1e}".format(p_val_1vs0)}')
    # plt.text(2.18, np.median(results_0vs1[data]-2), f'p-value: {"{:.1e}".format(p_val_0vs1)}')
    # plt.text(3.19, np.median(results_2vs0[data]-2), f'p-value: {"{:.1e}".format(p_val_2vs0)}')
    # plt.text(5.19, np.median(results_2vs1[data]-2), f'p-value: {"{:.1e}".format(p_val_2vs1)}')

    height = max([max(results_0vs0[data]), max(results_1vs0[data]), max(results_0vs1[data]), max(results_2vs0[data]), max(results_1vs1[data]), max(results_2vs1[data])])

    barplot_annotate_brackets([0,0,0,4], [1,2,3,5], [p_val_1vs0, p_val_0vs1, p_val_2vs0, p_val_2vs1], [1,2,3,4,5,6], [height]*6, dh=[.05, .1, .15, .05])

    plt.show()
