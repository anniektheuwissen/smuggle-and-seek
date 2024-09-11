import mesa
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from smuggle_and_seek.model import SmuggleAndSeekGame

"""
Run a batch run with specified parameters
"""

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

# Run:
params = {
    "k": 2,
    "l": range(2,5,1),
    "m": 5,
    "tom_customs": range(0,3,1),
    "tom_smuggler": range(0,3,1),
    "learning_speed1": 0.4,
    "learning_speed2": 0.2
}

results1 = mesa.batch_run(
    SmuggleAndSeekGame,
    parameters=params,
    iterations=1,
    display_progress=True,
)

params = {
    "k": 3,
    "l": 2,
    "m": 5,
    "tom_customs": range(0,3,1),
    "tom_smuggler": range(0,3,1),
    "learning_speed1": 0.4,
    "learning_speed2": 0.2
}

results2 = mesa.batch_run(
    SmuggleAndSeekGame,
    parameters=params,
    iterations=1,
    display_progress=True,
)


# Collect all data together
results_df1 = pd.DataFrame(results1)
results_df2 = pd.DataFrame(results2)
results_df = pd.concat([results_df1, results_df2])
results_df.to_csv('results.csv')


# Create graphs from csv:
results_2x2 = results_df[(results_df["k"] == 2) & (results_df["l"] == 2)]
results_2x2x2 = results_df[(results_df["k"] == 3) & (results_df["l"] == 2)]
results_3x3 = results_df[(results_df["k"] == 2) & (results_df["l"] == 3)]
results_4x4 = results_df[(results_df["k"] == 2) & (results_df["l"] == 4)]

results_2x2_0vs0 = results_2x2[(results_2x2["tom_customs"] == 0) & (results_2x2["tom_smuggler"] == 0)]
results_2x2_0vs1 = results_2x2[(results_2x2["tom_customs"] == 0) & (results_2x2["tom_smuggler"] == 1)]
results_2x2_0vs2 = results_2x2[(results_2x2["tom_customs"] == 0) & (results_2x2["tom_smuggler"] == 2)]
results_2x2_1vs0 = results_2x2[(results_2x2["tom_customs"] == 1) & (results_2x2["tom_smuggler"] == 0)]
results_2x2_1vs1 = results_2x2[(results_2x2["tom_customs"] == 1) & (results_2x2["tom_smuggler"] == 1)]
results_2x2_1vs2 = results_2x2[(results_2x2["tom_customs"] == 1) & (results_2x2["tom_smuggler"] == 2)]
results_2x2_2vs0 = results_2x2[(results_2x2["tom_customs"] == 2) & (results_2x2["tom_smuggler"] == 0)]
results_2x2_2vs1 = results_2x2[(results_2x2["tom_customs"] == 2) & (results_2x2["tom_smuggler"] == 1)]

results_2x2x2_0vs0 = results_2x2x2[(results_2x2x2["tom_customs"] == 0) & (results_2x2x2["tom_smuggler"] == 0)]
results_2x2x2_0vs1 = results_2x2x2[(results_2x2x2["tom_customs"] == 0) & (results_2x2x2["tom_smuggler"] == 1)]
results_2x2x2_0vs2 = results_2x2x2[(results_2x2x2["tom_customs"] == 0) & (results_2x2x2["tom_smuggler"] == 2)]
results_2x2x2_1vs0 = results_2x2x2[(results_2x2x2["tom_customs"] == 1) & (results_2x2x2["tom_smuggler"] == 0)]
results_2x2x2_1vs1 = results_2x2x2[(results_2x2x2["tom_customs"] == 1) & (results_2x2x2["tom_smuggler"] == 1)]
results_2x2x2_1vs2 = results_2x2x2[(results_2x2x2["tom_customs"] == 1) & (results_2x2x2["tom_smuggler"] == 2)]
results_2x2x2_2vs0 = results_2x2x2[(results_2x2x2["tom_customs"] == 2) & (results_2x2x2["tom_smuggler"] == 0)]
results_2x2x2_2vs1 = results_2x2x2[(results_2x2x2["tom_customs"] == 2) & (results_2x2x2["tom_smuggler"] == 1)]

results_3x3_0vs0 = results_3x3[(results_3x3["tom_customs"] == 0) & (results_3x3["tom_smuggler"] == 0)]
results_3x3_0vs1 = results_3x3[(results_3x3["tom_customs"] == 0) & (results_3x3["tom_smuggler"] == 1)]
results_3x3_0vs2 = results_3x3[(results_3x3["tom_customs"] == 0) & (results_3x3["tom_smuggler"] == 2)]
results_3x3_1vs0 = results_3x3[(results_3x3["tom_customs"] == 1) & (results_3x3["tom_smuggler"] == 0)]
results_3x3_1vs1 = results_3x3[(results_3x3["tom_customs"] == 1) & (results_3x3["tom_smuggler"] == 1)]
results_3x3_1vs2 = results_3x3[(results_3x3["tom_customs"] == 1) & (results_3x3["tom_smuggler"] == 2)]
results_3x3_2vs0 = results_3x3[(results_3x3["tom_customs"] == 2) & (results_3x3["tom_smuggler"] == 0)]
results_3x3_2vs1 = results_3x3[(results_3x3["tom_customs"] == 2) & (results_3x3["tom_smuggler"] == 1)]

results_4x4_0vs0 = results_4x4[(results_4x4["tom_customs"] == 0) & (results_4x4["tom_smuggler"] == 0)]
results_4x4_0vs1 = results_4x4[(results_4x4["tom_customs"] == 0) & (results_4x4["tom_smuggler"] == 1)]
results_4x4_0vs2 = results_4x4[(results_4x4["tom_customs"] == 0) & (results_4x4["tom_smuggler"] == 2)]
results_4x4_1vs0 = results_4x4[(results_4x4["tom_customs"] == 1) & (results_4x4["tom_smuggler"] == 0)]
results_4x4_1vs1 = results_4x4[(results_4x4["tom_customs"] == 1) & (results_4x4["tom_smuggler"] == 1)]
results_4x4_1vs2 = results_4x4[(results_4x4["tom_customs"] == 1) & (results_4x4["tom_smuggler"] == 2)]
results_4x4_2vs0 = results_4x4[(results_4x4["tom_customs"] == 2) & (results_4x4["tom_smuggler"] == 0)]
results_4x4_2vs1 = results_4x4[(results_4x4["tom_customs"] == 2) & (results_4x4["tom_smuggler"] == 1)]

for data in ["customs points", "smuggler points", "successful checks", "successful smuggles", "caught packages", "smuggled packages", "total checks", "total smuggles"]:
    
    t_stat_1vs0, p_val_1vs0 = stats.ttest_ind(results_2x2_0vs0[data], results_2x2_1vs0[data])
    t_stat_0vs1, p_val_0vs1 = stats.ttest_ind(results_2x2_0vs0[data], results_2x2_0vs1[data])
    t_stat_2vs0, p_val_2vs0 = stats.ttest_ind(results_2x2_0vs0[data], results_2x2_2vs0[data])
    t_stat_0vs2, p_val_0vs2 = stats.ttest_ind(results_2x2_0vs0[data], results_2x2_0vs2[data])
    t_stat_2vs1, p_val_2vs1 = stats.ttest_ind(results_2x2_1vs1[data], results_2x2_2vs1[data])
    t_stat_1vs2, p_val_1vs2 = stats.ttest_ind(results_2x2_1vs1[data], results_2x2_1vs2[data])
    
    fig, ax = plt.subplots(figsize=(16,8))
    boxplot = ax.boxplot([results_2x2_0vs0[data], results_2x2_1vs0[data], results_2x2_0vs1[data], results_2x2_2vs0[data], results_2x2_0vs2[data], results_2x2_1vs1[data], results_2x2_2vs1[data], results_2x2_1vs2[data]],
                            labels=['ToM0 customs vs\n ToM0 smuggler', 'ToM1 customs vs\n ToM0 smuggler', 'ToM0 customs vs\n ToM1 smuggler', 'ToM2 customs vs\n ToM0 smuggler', 
                                    'ToM0 customs vs\n ToM2 smuggler', 'ToM1 customs vs\n ToM1 smuggler', 'ToM2 customs vs\n ToM1 smuggler', 'ToM1 customs vs\n ToM2 smuggler'], patch_artist=True, medianprops={'color': 'black'}
                            )
    colors = ['#A8A9AD', '#728FCE', '#728FCE', '#728FCE', '#728FCE', '#A8A9AD', '#728FCE', '#728FCE']
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)
    plt.ylabel(data)
    plt.title(f"Number of {data} after 365 days")

    height = max([max(results_2x2_0vs0[data]), max(results_2x2_1vs0[data]), max(results_2x2_0vs1[data]), max(results_2x2_2vs0[data]), max(results_2x2_0vs2[data]), max(results_2x2_1vs1[data]), max(results_2x2_2vs1[data]), max(results_2x2_1vs2[data])])
    barplot_annotate_brackets([0,0,0,0,5,5], [1,2,3,4,6,7], [p_val_1vs0, p_val_0vs1, p_val_2vs0, p_val_0vs2, p_val_2vs1, p_val_1vs2], [1,2,3,4,5,6,7,8], [height]*8, dh=[.05, .1, .15, .2, .05, .1])

    # plt.savefig("results/2x2_"+str(data)+".png")
    plt.show()


    t_stat_1vs0, p_val_1vs0 = stats.ttest_ind(results_2x2x2_0vs0[data], results_2x2x2_1vs0[data])
    t_stat_0vs1, p_val_0vs1 = stats.ttest_ind(results_2x2x2_0vs0[data], results_2x2x2_0vs1[data])
    t_stat_2vs0, p_val_2vs0 = stats.ttest_ind(results_2x2x2_0vs0[data], results_2x2x2_2vs0[data])
    t_stat_0vs2, p_val_0vs2 = stats.ttest_ind(results_2x2x2_0vs0[data], results_2x2x2_0vs2[data])
    t_stat_2vs1, p_val_2vs1 = stats.ttest_ind(results_2x2x2_1vs1[data], results_2x2x2_2vs1[data])
    t_stat_1vs2, p_val_1vs2 = stats.ttest_ind(results_2x2x2_1vs1[data], results_2x2x2_1vs2[data])
    
    fig, ax = plt.subplots(figsize=(16,8))
    boxplot = ax.boxplot([results_2x2x2_0vs0[data], results_2x2x2_1vs0[data], results_2x2x2_0vs1[data], results_2x2x2_2vs0[data], results_2x2x2_0vs2[data], results_2x2x2_1vs1[data], results_2x2x2_2vs1[data], results_2x2x2_1vs2[data]],
                            labels=['ToM0 customs vs\n ToM0 smuggler', 'ToM1 customs vs\n ToM0 smuggler', 'ToM0 customs vs\n ToM1 smuggler', 'ToM2 customs vs\n ToM0 smuggler', 
                                    'ToM0 customs vs\n ToM2 smuggler', 'ToM1 customs vs\n ToM1 smuggler', 'ToM2 customs vs\n ToM1 smuggler', 'ToM1 customs vs\n ToM2 smuggler'], patch_artist=True, medianprops={'color': 'black'}
                            )
    colors = ['#A8A9AD', '#728FCE', '#728FCE', '#728FCE', '#728FCE', '#A8A9AD', '#728FCE', '#728FCE']
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)
    plt.ylabel(data)
    plt.title(f"Number of {data} after 365 days")

    height = max([max(results_2x2x2_0vs0[data]), max(results_2x2x2_1vs0[data]), max(results_2x2x2_0vs1[data]), max(results_2x2x2_2vs0[data]), max(results_2x2x2_0vs2[data]), max(results_2x2x2_1vs1[data]), max(results_2x2x2_2vs1[data]), max(results_2x2x2_1vs2[data])])
    barplot_annotate_brackets([0,0,0,0,5,5], [1,2,3,4,6,7], [p_val_1vs0, p_val_0vs1, p_val_2vs0, p_val_0vs2, p_val_2vs1, p_val_1vs2], [1,2,3,4,5,6,7,8], [height]*8, dh=[.05, .1, .15, .2, .05, .1])

    # plt.savefig("results/2x2x2_+str(data)+".png")
    plt.show()


    t_stat_1vs0, p_val_1vs0 = stats.ttest_ind(results_3x3_0vs0[data], results_3x3_1vs0[data])
    t_stat_0vs1, p_val_0vs1 = stats.ttest_ind(results_3x3_0vs0[data], results_3x3_0vs1[data])
    t_stat_2vs0, p_val_2vs0 = stats.ttest_ind(results_3x3_0vs0[data], results_3x3_2vs0[data])
    t_stat_0vs2, p_val_0vs2 = stats.ttest_ind(results_3x3_0vs0[data], results_3x3_0vs2[data])
    t_stat_2vs1, p_val_2vs1 = stats.ttest_ind(results_3x3_1vs1[data], results_3x3_2vs1[data])
    t_stat_1vs2, p_val_1vs2 = stats.ttest_ind(results_3x3_1vs1[data], results_3x3_1vs2[data])
    
    fig, ax = plt.subplots(figsize=(16,8))
    boxplot = ax.boxplot([results_3x3_0vs0[data], results_3x3_1vs0[data], results_3x3_0vs1[data], results_3x3_2vs0[data], results_3x3_0vs2[data], results_3x3_1vs1[data], results_3x3_2vs1[data], results_3x3_1vs2[data]],
                            labels=['ToM0 customs vs\n ToM0 smuggler', 'ToM1 customs vs\n ToM0 smuggler', 'ToM0 customs vs\n ToM1 smuggler', 'ToM2 customs vs\n ToM0 smuggler', 
                                    'ToM0 customs vs\n ToM2 smuggler', 'ToM1 customs vs\n ToM1 smuggler', 'ToM2 customs vs\n ToM1 smuggler', 'ToM1 customs vs\n ToM2 smuggler'], patch_artist=True, medianprops={'color': 'black'}
                            )
    colors = ['#A8A9AD', '#728FCE', '#728FCE', '#728FCE', '#728FCE', '#A8A9AD', '#728FCE', '#728FCE']
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)
    plt.ylabel(data)
    plt.title(f"Number of {data} after 365 days")

    height = max([max(results_3x3_0vs0[data]), max(results_3x3_1vs0[data]), max(results_3x3_0vs1[data]), max(results_3x3_2vs0[data]), max(results_3x3_0vs2[data]), max(results_3x3_1vs1[data]), max(results_3x3_2vs1[data]), max(results_3x3_1vs2[data])])
    barplot_annotate_brackets([0,0,0,0,5,5], [1,2,3,4,6,7], [p_val_1vs0, p_val_0vs1, p_val_2vs0, p_val_0vs2, p_val_2vs1, p_val_1vs2], [1,2,3,4,5,6,7,8], [height]*8, dh=[.05, .1, .15, .2, .05, .1])

    # plt.savefig("results/3x3_+str(data)+".png")
    plt.show()


    t_stat_1vs0, p_val_1vs0 = stats.ttest_ind(results_4x4_0vs0[data], results_4x4_1vs0[data])
    t_stat_0vs1, p_val_0vs1 = stats.ttest_ind(results_4x4_0vs0[data], results_4x4_0vs1[data])
    t_stat_2vs0, p_val_2vs0 = stats.ttest_ind(results_4x4_0vs0[data], results_4x4_2vs0[data])
    t_stat_0vs2, p_val_0vs2 = stats.ttest_ind(results_4x4_0vs0[data], results_4x4_0vs2[data])
    t_stat_2vs1, p_val_2vs1 = stats.ttest_ind(results_4x4_1vs1[data], results_4x4_2vs1[data])
    t_stat_1vs2, p_val_1vs2 = stats.ttest_ind(results_4x4_1vs1[data], results_4x4_1vs2[data])
    
    fig, ax = plt.subplots(figsize=(16,8))
    boxplot = ax.boxplot([results_4x4_0vs0[data], results_4x4_1vs0[data], results_4x4_0vs1[data], results_4x4_2vs0[data], results_4x4_0vs2[data], results_4x4_1vs1[data], results_4x4_2vs1[data], results_4x4_1vs2[data]],
                            labels=['ToM0 customs vs\n ToM0 smuggler', 'ToM1 customs vs\n ToM0 smuggler', 'ToM0 customs vs\n ToM1 smuggler', 'ToM2 customs vs\n ToM0 smuggler', 
                                    'ToM0 customs vs\n ToM2 smuggler', 'ToM1 customs vs\n ToM1 smuggler', 'ToM2 customs vs\n ToM1 smuggler', 'ToM1 customs vs\n ToM2 smuggler'], patch_artist=True, medianprops={'color': 'black'}
                            )
    colors = ['#A8A9AD', '#728FCE', '#728FCE', '#728FCE', '#728FCE', '#A8A9AD', '#728FCE', '#728FCE']
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)
    plt.ylabel(data)
    plt.title(f"Number of {data} after 365 days")

    height = max([max(results_4x4_0vs0[data]), max(results_4x4_1vs0[data]), max(results_4x4_0vs1[data]), max(results_4x4_2vs0[data]), max(results_4x4_0vs2[data]), max(results_4x4_1vs1[data]), max(results_4x4_2vs1[data]), max(results_4x4_1vs2[data])])
    barplot_annotate_brackets([0,0,0,0,5,5], [1,2,3,4,6,7], [p_val_1vs0, p_val_0vs1, p_val_2vs0, p_val_0vs2, p_val_2vs1, p_val_1vs2], [1,2,3,4,5,6,7,8], [height]*8, dh=[.05, .1, .15, .2, .05, .1])

    # plt.savefig("results/4x4_+str(data)+".png")
    plt.show()


increase_2x2_0vs1 = {}; increase_2x2x2_0vs1 = {}; increase_3x3_0vs1 = {}; increase_4x4_0vs1 = {}
increase_2x2_1vs0 = {}; increase_2x2x2_1vs0 = {}; increase_3x3_1vs0 = {}; increase_4x4_1vs0 = {}
increase_2x2_1vs2 = {}; increase_2x2x2_1vs2 = {}; increase_3x3_1vs2 = {}; increase_4x4_1vs2 = {}
increase_2x2_2vs1 = {}; increase_2x2x2_2vs1 = {}; increase_3x3_2vs1 = {}; increase_4x4_2vs1 = {}
for data in ["customs points", "smuggler points", "successful checks", "successful smuggles", "caught packages", "smuggled packages", "total checks", "total smuggles"]:
    increase_2x2_0vs1[data] = results_2x2_0vs1[data].values - results_2x2_0vs0[data].values
    increase_2x2x2_0vs1[data] = results_2x2x2_0vs1[data].values - results_2x2x2_0vs0[data].values
    increase_3x3_0vs1[data] = results_3x3_0vs1[data].values - results_3x3_0vs0[data].values
    increase_4x4_0vs1[data] = results_4x4_0vs1[data].values - results_4x4_0vs0[data].values

    increase_2x2_1vs0[data] = results_2x2_1vs0[data].values - results_2x2_0vs0[data].values
    increase_2x2x2_1vs0[data] = results_2x2x2_1vs0[data].values - results_2x2x2_0vs0[data].values
    increase_3x3_1vs0[data] = results_3x3_1vs0[data].values - results_3x3_0vs0[data].values
    increase_4x4_1vs0[data] = results_4x4_1vs0[data].values - results_4x4_0vs0[data].values

    increase_2x2_1vs2[data] = results_2x2_1vs2[data].values - results_2x2_1vs1[data].values
    increase_2x2x2_1vs2[data] = results_2x2x2_1vs2[data].values - results_2x2x2_1vs1[data].values
    increase_3x3_1vs2[data] = results_3x3_1vs2[data].values - results_3x3_1vs1[data].values
    increase_4x4_1vs2[data] = results_4x4_1vs2[data].values - results_4x4_1vs1[data].values

    increase_2x2_2vs1[data] = results_2x2_2vs1[data].values - results_2x2_1vs1[data].values
    increase_2x2x2_2vs1[data] = results_2x2x2_2vs1[data].values - results_2x2x2_1vs1[data].values
    increase_3x3_2vs1[data] = results_3x3_2vs1[data].values - results_3x3_1vs1[data].values
    increase_4x4_2vs1[data] = results_4x4_2vs1[data].values - results_4x4_1vs1[data].values


for data in ["customs points", "smuggler points", "successful checks", "successful smuggles", "caught packages", "smuggled packages", "total checks", "total smuggles"]:
    fig, ax = plt.subplots(figsize=(16,8))
    boxplot = ax.boxplot([increase_2x2_0vs1[data], increase_2x2x2_0vs1[data], increase_3x3_0vs1[data], increase_4x4_0vs1[data]],
                            positions=[4,8,9,16], patch_artist=True, medianprops={'color': 'black'}
                            )
    colors = ['#728FCE', '#728FCE', '#728FCE', '#728FCE']
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)
    
    # Set x-axis limits and ticks
    plt.xlim(3, 17)  # Set x-axis limits to accommodate space around boxplots
    plt.xticks(range(4, 17))  # Set x-axis ticks for 16 positions

    # Optionally, add tick labels (if you want specific labels, use the second parameter)
    plt.gca().set_xticklabels(range(4, 17))

    means = [np.mean(increase_2x2_0vs1[data]), np.mean(increase_2x2x2_0vs1[data]), np.mean(increase_3x3_0vs1[data]), np.mean(increase_4x4_0vs1[data])]
    x = [4,8,9,16]
    a, b = np.polyfit(x, means, 1)

    plt.plot(range(4,17), a*range(4,17)+b, ls = '--')

    plt.ylabel(f"Increase of {data}")
    plt.xlabel("Number of container types")
    plt.title(f"Increase of {data} after 365 days when ToM0 smugglers becomes ToM1 smuggler against ToM0 customs")

    # plt.savefig("results/increase_0vs1_"+str(data)+".png")
    plt.show()


    fig, ax = plt.subplots(figsize=(16,8))
    boxplot = ax.boxplot([increase_2x2_1vs0[data], increase_2x2x2_1vs0[data], increase_3x3_1vs0[data], increase_4x4_1vs0[data]],
                            positions=[4,8,9,16], patch_artist=True, medianprops={'color': 'black'}
                            )
    colors = ['#728FCE', '#728FCE', '#728FCE', '#728FCE']
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)

    # Set x-axis limits and ticks
    plt.xlim(3, 17)  # Set x-axis limits to accommodate space around boxplots
    plt.xticks(range(4, 17))  # Set x-axis ticks for 16 positions

    # Optionally, add tick labels (if you want specific labels, use the second parameter)
    plt.gca().set_xticklabels(range(4, 17))
        
    means = [np.mean(increase_2x2_1vs0[data]), np.mean(increase_2x2x2_1vs0[data]), np.mean(increase_3x3_1vs0[data]), np.mean(increase_4x4_1vs0[data])]
    x = [4,8,9,16]
    a, b = np.polyfit(x, means, 1)

    plt.plot(range(4,17), a*range(4,17)+b, ls = '--')

    plt.ylabel(f"Increase of {data}")
    plt.ylabel(data)
    plt.xlabel("Number of container types")
    plt.title(f"Increase of {data} after 365 days when ToM0 customs becomes ToM1 customs against ToM0 smuggler")

    # plt.savefig("results/increase_1vs0_"+str(data)+".png")
    plt.show()


    fig, ax = plt.subplots(figsize=(16,8))
    boxplot = ax.boxplot([increase_2x2_1vs2[data], increase_2x2x2_1vs2[data], increase_3x3_1vs2[data], increase_4x4_1vs2[data]],
                            positions=[4,8,9,16], patch_artist=True, medianprops={'color': 'black'}
                            )
    colors = ['#728FCE', '#728FCE', '#728FCE', '#728FCE']
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)
    
    # Set x-axis limits and ticks
    plt.xlim(3, 17)  # Set x-axis limits to accommodate space around boxplots
    plt.xticks(range(4, 17))  # Set x-axis ticks for 16 positions

    # Optionally, add tick labels (if you want specific labels, use the second parameter)
    plt.gca().set_xticklabels(range(4, 17))

    means = [np.mean(increase_2x2_1vs2[data]), np.mean(increase_2x2x2_1vs2[data]), np.mean(increase_3x3_1vs2[data]), np.mean(increase_4x4_1vs2[data])]
    x = [4,8,9,16]
    a, b = np.polyfit(x, means, 1)

    plt.plot(range(4,17), a*range(4,17)+b, ls = '--')

    plt.ylabel(f"Increase of {data}")
    plt.xlabel("Number of container types")
    plt.title(f"Increase of {data} after 365 days when ToM1 smuggler becomes ToM2 smuggler against ToM1 customs")

    # plt.savefig("results/increase_1vs2_"+str(data)+".png")
    plt.show()
    

    fig, ax = plt.subplots(figsize=(16,8))
    boxplot = ax.boxplot([increase_2x2_2vs1[data], increase_2x2x2_2vs1[data], increase_3x3_2vs1[data], increase_4x4_2vs1[data]],
                            positions=[4,8,9,16], patch_artist=True, medianprops={'color': 'black'}
                            )
    colors = ['#728FCE', '#728FCE', '#728FCE', '#728FCE']
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)

    # Set x-axis limits and ticks
    plt.xlim(3, 17)  # Set x-axis limits to accommodate space around boxplots
    plt.xticks(range(4, 17))  # Set x-axis ticks for 16 positions

    # Optionally, add tick labels (if you want specific labels, use the second parameter)
    plt.gca().set_xticklabels(range(4, 17))
    
    means = [np.mean(increase_2x2_2vs1[data]), np.mean(increase_2x2x2_2vs1[data]), np.mean(increase_3x3_2vs1[data]), np.mean(increase_4x4_2vs1[data])]
    x = [4,8,9,16]
    a, b = np.polyfit(x, means, 1)

    plt.plot(range(4,17), a*range(4,17)+b, ls = '--')

    plt.ylabel(f"Increase of {data}")
    plt.xlabel("Number of container types")
    plt.title(f"Increase of {data} after 365 days when ToM1 customs becomes ToM2 customs against ToM1 smuggler")

    # plt.savefig("results/increase_2vs1_"+str(data)+".png")
    plt.show()


percentage_increase_2x2_0vs1 = {}; percentage_increase_2x2x2_0vs1 = {}; percentage_increase_3x3_0vs1 = {}; percentage_increase_4x4_0vs1 = {}
percentage_increase_2x2_1vs0 = {}; percentage_increase_2x2x2_1vs0 = {}; percentage_increase_3x3_1vs0 = {}; percentage_increase_4x4_1vs0 = {}
percentage_increase_2x2_1vs2 = {}; percentage_increase_2x2x2_1vs2 = {}; percentage_increase_3x3_1vs2 = {}; percentage_increase_4x4_1vs2 = {}
percentage_increase_2x2_2vs1 = {}; percentage_increase_2x2x2_2vs1 = {}; percentage_increase_3x3_2vs1 = {}; percentage_increase_4x4_2vs1 = {}
for data in ["customs points", "smuggler points", "successful checks", "successful smuggles", "caught packages", "smuggled packages", "total checks", "total smuggles"]:
    percentage_increase_2x2_0vs1[data] = (results_2x2_0vs1[data].values - results_2x2_0vs0[data].values) / results_2x2_0vs1[data].values
    percentage_increase_2x2x2_0vs1[data] = (results_2x2x2_0vs1[data].values - results_2x2x2_0vs0[data].values) / results_2x2x2_0vs1[data].values
    percentage_increase_3x3_0vs1[data] = (results_3x3_0vs1[data].values - results_3x3_0vs0[data].values) / results_3x3_0vs1[data].values
    percentage_increase_4x4_0vs1[data] = (results_4x4_0vs1[data].values - results_4x4_0vs0[data].values) / results_4x4_0vs1[data].values

    percentage_increase_2x2_1vs0[data] = (results_2x2_1vs0[data].values - results_2x2_0vs0[data].values) / results_2x2_1vs0[data].values
    percentage_increase_2x2x2_1vs0[data] = (results_2x2x2_1vs0[data].values - results_2x2x2_0vs0[data].values) / results_2x2x2_1vs0[data].values
    percentage_increase_3x3_1vs0[data] = (results_3x3_1vs0[data].values - results_3x3_0vs0[data].values) / results_3x3_1vs0[data].values
    percentage_increase_4x4_1vs0[data] = (results_4x4_1vs0[data].values - results_4x4_0vs0[data].values) / results_4x4_1vs0[data].values

    percentage_increase_2x2_1vs2[data] = (results_2x2_1vs2[data].values - results_2x2_1vs1[data].values) / results_2x2_1vs2[data].values
    percentage_increase_2x2x2_1vs2[data] = (results_2x2x2_1vs2[data].values - results_2x2x2_1vs1[data].values) / results_2x2x2_1vs2[data].values
    percentage_increase_3x3_1vs2[data] = (results_3x3_1vs2[data].values - results_3x3_1vs1[data].values) / results_3x3_1vs2[data].values
    percentage_increase_4x4_1vs2[data] = (results_4x4_1vs2[data].values - results_4x4_1vs1[data].values) / results_4x4_1vs2[data].values

    percentage_increase_2x2_2vs1[data] = (results_2x2_2vs1[data].values - results_2x2_1vs1[data].values) / results_2x2_2vs1[data].values
    percentage_increase_2x2x2_2vs1[data] = (results_2x2x2_2vs1[data].values - results_2x2x2_1vs1[data].values) / results_2x2x2_2vs1[data].values
    percentage_increase_3x3_2vs1[data] = (results_3x3_2vs1[data].values - results_3x3_1vs1[data].values) / results_3x3_2vs1[data].values
    percentage_increase_4x4_2vs1[data] = (results_4x4_2vs1[data].values - results_4x4_1vs1[data].values) / results_4x4_2vs1[data].values


for data in ["customs points", "smuggler points", "successful checks", "successful smuggles", "caught packages", "smuggled packages", "total checks", "total smuggles"]:
    fig, ax = plt.subplots(figsize=(16,8))
    boxplot = ax.boxplot([percentage_increase_2x2_0vs1[data], percentage_increase_2x2x2_0vs1[data], percentage_increase_3x3_0vs1[data], percentage_increase_4x4_0vs1[data]],
                            positions=[4,8,9,16], patch_artist=True, medianprops={'color': 'black'}
                            )
    colors = ['#728FCE', '#728FCE', '#728FCE', '#728FCE']
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)

    # Set x-axis limits and ticks
    plt.xlim(3, 17)  # Set x-axis limits to accommodate space around boxplots
    plt.xticks(range(4, 17))  # Set x-axis ticks for 16 positions

    # Optionally, add tick labels (if you want specific labels, use the second parameter)
    plt.gca().set_xticklabels(range(4, 17))

    means = [np.mean(percentage_increase_2x2_0vs1[data]), np.mean(percentage_increase_2x2x2_0vs1[data]), np.mean(percentage_increase_3x3_0vs1[data]), np.mean(percentage_increase_4x4_0vs1[data])]
    x = [4,8,9,16]
    a, b = np.polyfit(x, means, 1)

    plt.plot(range(4,17), a*range(4,17)+b, ls = '--')

    plt.ylabel(f"Percentage increase of {data}")
    plt.xlabel("Number of container types")
    plt.title(f"Percentage increase of {data} after 365 days when ToM0 smugglers becomes ToM1 smuggler against ToM0 customs")

    # plt.savefig("results/percentageincrease_0vs1_"+str(data)+".png")
    plt.show()


    fig, ax = plt.subplots(figsize=(16,8))
    boxplot = ax.boxplot([percentage_increase_2x2_1vs0[data], percentage_increase_2x2x2_1vs0[data], percentage_increase_3x3_1vs0[data], percentage_increase_4x4_1vs0[data]],
                            positions=[4,8,9,16], patch_artist=True, medianprops={'color': 'black'}
                            )
    colors = ['#728FCE', '#728FCE', '#728FCE', '#728FCE']
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)

    # Set x-axis limits and ticks
    plt.xlim(3, 17)  # Set x-axis limits to accommodate space around boxplots
    plt.xticks(range(4, 17))  # Set x-axis ticks for 16 positions

    # Optionally, add tick labels (if you want specific labels, use the second parameter)
    plt.gca().set_xticklabels(range(4, 17))

    means = [np.mean(percentage_increase_2x2_1vs0[data]), np.mean(percentage_increase_2x2x2_1vs0[data]), np.mean(percentage_increase_3x3_1vs0[data]), np.mean(percentage_increase_4x4_1vs0[data])]
    x = [4,8,9,16]
    a, b = np.polyfit(x, means, 1)

    plt.plot(range(4,17), a*range(4,17)+b, ls = '--')

    plt.ylabel(f"Percentage increase of {data}")
    plt.xlabel("Number of container types")
    plt.title(f"Percentage increase of {data} after 365 days when ToM0 customs becomes ToM1 customs against ToM0 smuggler")

    # plt.savefig("results/percentageincrease_1vs0_"+str(data)+".png")
    plt.show()


    fig, ax = plt.subplots(figsize=(16,8))
    boxplot = ax.boxplot([percentage_increase_2x2_1vs2[data], percentage_increase_2x2x2_1vs2[data], percentage_increase_3x3_1vs2[data], percentage_increase_4x4_1vs2[data]],
                            labels=[4,8,9,16], patch_artist=True, medianprops={'color': 'black'}
                            )
    colors = ['#728FCE', '#728FCE', '#728FCE', '#728FCE']
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)
    
    # Set x-axis limits and ticks
    plt.xlim(3, 17)  # Set x-axis limits to accommodate space around boxplots
    plt.xticks(range(4, 17))  # Set x-axis ticks for 16 positions

    # Optionally, add tick labels (if you want specific labels, use the second parameter)
    plt.gca().set_xticklabels(range(4, 17))

    means = [np.mean(percentage_increase_2x2_1vs2[data]), np.mean(percentage_increase_2x2x2_1vs2[data]), np.mean(percentage_increase_3x3_1vs2[data]), np.mean(percentage_increase_4x4_1vs2[data])]
    x = [4,8,9,16]
    a, b = np.polyfit(x, means, 1)

    plt.plot(range(4,17), a*range(4,17)+b, ls = '--')

    plt.ylabel(f"Percentage increase of {data}")
    plt.xlabel("Number of container types")
    plt.title(f"Percentage increase of {data} after 365 days when ToM1 smuggler becomes ToM2 smuggler against ToM1 customs")

    # plt.savefig("results/percentageincrease_1vs2_"+str(data)+".png")
    plt.show()


    fig, ax = plt.subplots(figsize=(16,8))
    boxplot = ax.boxplot([percentage_increase_2x2_2vs1[data], percentage_increase_2x2x2_2vs1[data], percentage_increase_3x3_2vs1[data], percentage_increase_4x4_2vs1[data]],
                            labels=[4,8,9,16], patch_artist=True, medianprops={'color': 'black'}
                            )
    colors = ['#728FCE', '#728FCE', '#728FCE', '#728FCE']
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)
    
    # Set x-axis limits and ticks
    plt.xlim(3, 17)  # Set x-axis limits to accommodate space around boxplots
    plt.xticks(range(4, 17))  # Set x-axis ticks for 16 positions

    # Optionally, add tick labels (if you want specific labels, use the second parameter)
    plt.gca().set_xticklabels(range(4, 17))

    means = [np.mean(percentage_increase_2x2_2vs1[data]), np.mean(percentage_increase_2x2x2_2vs1[data]), np.mean(percentage_increase_3x3_2vs1[data]), np.mean(percentage_increase_4x4_2vs1[data])]
    x = [4,8,9,16]
    a, b = np.polyfit(x, means, 1)

    plt.plot(range(4,17), a*range(4,17)+b, ls = '--')

    plt.ylabel(f"Percentage increase of {data}")
    plt.xlabel("Number of container types")
    plt.title(f"Percentage increase of {data} after 365 days when ToM1 customs becomes ToM2 customs against ToM1 smuggler")

    # plt.savefig("results/percentageincrease_2vs1_"+str(data)+".png")
    plt.show()