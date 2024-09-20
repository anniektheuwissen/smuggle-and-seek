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
    "r_s": 2,
    "r_c": 2,
    "cc_s": 6,
    "cc_c": 6,
    "fc_s": 1,
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
    "k": range(3,5,1),
    "l": 2,
    "m": 5,
    "r_s": 2,
    "r_c": 2,
    "cc_s": 6,
    "cc_c": 6,
    "fc_s": 1,
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
results_2x2x2x2 = results_df[(results_df["k"] == 4) & (results_df["l"] == 2)]
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

results_2x2x2x2_0vs0 = results_2x2x2x2[(results_2x2x2x2["tom_customs"] == 0) & (results_2x2x2x2["tom_smuggler"] == 0)]
results_2x2x2x2_0vs1 = results_2x2x2x2[(results_2x2x2x2["tom_customs"] == 0) & (results_2x2x2x2["tom_smuggler"] == 1)]
results_2x2x2x2_0vs2 = results_2x2x2x2[(results_2x2x2x2["tom_customs"] == 0) & (results_2x2x2x2["tom_smuggler"] == 2)]
results_2x2x2x2_1vs0 = results_2x2x2x2[(results_2x2x2x2["tom_customs"] == 1) & (results_2x2x2x2["tom_smuggler"] == 0)]
results_2x2x2x2_1vs1 = results_2x2x2x2[(results_2x2x2x2["tom_customs"] == 1) & (results_2x2x2x2["tom_smuggler"] == 1)]
results_2x2x2x2_1vs2 = results_2x2x2x2[(results_2x2x2x2["tom_customs"] == 1) & (results_2x2x2x2["tom_smuggler"] == 2)]
results_2x2x2x2_2vs0 = results_2x2x2x2[(results_2x2x2x2["tom_customs"] == 2) & (results_2x2x2x2["tom_smuggler"] == 0)]
results_2x2x2x2_2vs1 = results_2x2x2x2[(results_2x2x2x2["tom_customs"] == 2) & (results_2x2x2x2["tom_smuggler"] == 1)]

results_4x4_0vs0 = results_4x4[(results_4x4["tom_customs"] == 0) & (results_4x4["tom_smuggler"] == 0)]
results_4x4_0vs1 = results_4x4[(results_4x4["tom_customs"] == 0) & (results_4x4["tom_smuggler"] == 1)]
results_4x4_0vs2 = results_4x4[(results_4x4["tom_customs"] == 0) & (results_4x4["tom_smuggler"] == 2)]
results_4x4_1vs0 = results_4x4[(results_4x4["tom_customs"] == 1) & (results_4x4["tom_smuggler"] == 0)]
results_4x4_1vs1 = results_4x4[(results_4x4["tom_customs"] == 1) & (results_4x4["tom_smuggler"] == 1)]
results_4x4_1vs2 = results_4x4[(results_4x4["tom_customs"] == 1) & (results_4x4["tom_smuggler"] == 2)]
results_4x4_2vs0 = results_4x4[(results_4x4["tom_customs"] == 2) & (results_4x4["tom_smuggler"] == 0)]
results_4x4_2vs1 = results_4x4[(results_4x4["tom_customs"] == 2) & (results_4x4["tom_smuggler"] == 1)]

plt.rcParams.update({'font.size': 24})

for data in ["customs points", "smuggler points", "successful checks", "successful smuggles", "caught packages", "smuggled packages", "total checks", "total smuggles"]:

    t_stat_1vs0, p_val_1vs0 = stats.ttest_ind(results_2x2_0vs0[data], results_2x2_1vs0[data])
    t_stat_0vs1, p_val_0vs1 = stats.ttest_ind(results_2x2_0vs0[data], results_2x2_0vs1[data])
    t_stat_2vs0, p_val_2vs0 = stats.ttest_ind(results_2x2_0vs0[data], results_2x2_2vs0[data])
    t_stat_0vs2, p_val_0vs2 = stats.ttest_ind(results_2x2_0vs0[data], results_2x2_0vs2[data])
    t_stat_2vs1, p_val_2vs1 = stats.ttest_ind(results_2x2_1vs1[data], results_2x2_2vs1[data])
    t_stat_1vs2, p_val_1vs2 = stats.ttest_ind(results_2x2_1vs1[data], results_2x2_1vs2[data])
    
    fig, ax = plt.subplots(figsize=(26,13))
    boxplot = ax.boxplot([results_2x2_0vs0[data], results_2x2_1vs0[data], results_2x2_0vs1[data], results_2x2_2vs0[data], results_2x2_0vs2[data], results_2x2_1vs1[data], results_2x2_2vs1[data], results_2x2_1vs2[data]],
                            labels=['ToM0 customs vs\n ToM0 smuggler', 'ToM1 customs vs\n ToM0 smuggler', 'ToM0 customs vs\n ToM1 smuggler', 'ToM2 customs vs\n ToM0 smuggler', 
                                    'ToM0 customs vs\n ToM2 smuggler', 'ToM1 customs vs\n ToM1 smuggler', 'ToM2 customs vs\n ToM1 smuggler', 'ToM1 customs vs\n ToM2 smuggler'], patch_artist=True, medianprops={'color': 'black'}
                            )
    colors = ['#A8A9AD', '#728FCE', '#728FCE', '#728FCE', '#728FCE', '#A8A9AD', '#728FCE', '#728FCE']
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)
    plt.ylabel(data, fontsize=30)
    plt.tick_params(axis='y', labelsize=30)
    # plt.title(f"Number of {data} after 365 days in game environment 1")

    height = max([max(results_2x2_0vs0[data]), max(results_2x2_1vs0[data]), max(results_2x2_0vs1[data]), max(results_2x2_2vs0[data]), max(results_2x2_0vs2[data]), max(results_2x2_1vs1[data]), max(results_2x2_2vs1[data]), max(results_2x2_1vs2[data])])
    barplot_annotate_brackets([0,0,0,0,5,5], [1,2,3,4,6,7], [p_val_1vs0, p_val_0vs1, p_val_2vs0, p_val_0vs2, p_val_2vs1, p_val_1vs2], [1,2,3,4,5,6,7,8], [height]*8, dh=[.05, .1, .15, .2, .05, .1])

    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.995)
    plt.savefig("results/2x2_"+str(data)+".png")
    # plt.show()


    t_stat_1vs0, p_val_1vs0 = stats.ttest_ind(results_2x2x2_0vs0[data], results_2x2x2_1vs0[data])
    t_stat_0vs1, p_val_0vs1 = stats.ttest_ind(results_2x2x2_0vs0[data], results_2x2x2_0vs1[data])
    t_stat_2vs0, p_val_2vs0 = stats.ttest_ind(results_2x2x2_0vs0[data], results_2x2x2_2vs0[data])
    t_stat_0vs2, p_val_0vs2 = stats.ttest_ind(results_2x2x2_0vs0[data], results_2x2x2_0vs2[data])
    t_stat_2vs1, p_val_2vs1 = stats.ttest_ind(results_2x2x2_1vs1[data], results_2x2x2_2vs1[data])
    t_stat_1vs2, p_val_1vs2 = stats.ttest_ind(results_2x2x2_1vs1[data], results_2x2x2_1vs2[data])
    
    fig, ax = plt.subplots(figsize=(26,13))
    boxplot = ax.boxplot([results_2x2x2_0vs0[data], results_2x2x2_1vs0[data], results_2x2x2_0vs1[data], results_2x2x2_2vs0[data], results_2x2x2_0vs2[data], results_2x2x2_1vs1[data], results_2x2x2_2vs1[data], results_2x2x2_1vs2[data]],
                            labels=['ToM0 customs vs\n ToM0 smuggler', 'ToM1 customs vs\n ToM0 smuggler', 'ToM0 customs vs\n ToM1 smuggler', 'ToM2 customs vs\n ToM0 smuggler', 
                                    'ToM0 customs vs\n ToM2 smuggler', 'ToM1 customs vs\n ToM1 smuggler', 'ToM2 customs vs\n ToM1 smuggler', 'ToM1 customs vs\n ToM2 smuggler'], patch_artist=True, medianprops={'color': 'black'}
                            )
    colors = ['#A8A9AD', '#728FCE', '#728FCE', '#728FCE', '#728FCE', '#A8A9AD', '#728FCE', '#728FCE']
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)
    plt.ylabel(data, fontsize=30)
    plt.tick_params(axis='y', labelsize=30)
    # plt.title(f"Number of {data} after 365 days in game environment 2")

    height = max([max(results_2x2x2_0vs0[data]), max(results_2x2x2_1vs0[data]), max(results_2x2x2_0vs1[data]), max(results_2x2x2_2vs0[data]), max(results_2x2x2_0vs2[data]), max(results_2x2x2_1vs1[data]), max(results_2x2x2_2vs1[data]), max(results_2x2x2_1vs2[data])])
    barplot_annotate_brackets([0,0,0,0,5,5], [1,2,3,4,6,7], [p_val_1vs0, p_val_0vs1, p_val_2vs0, p_val_0vs2, p_val_2vs1, p_val_1vs2], [1,2,3,4,5,6,7,8], [height]*8, dh=[.05, .1, .15, .2, .05, .1])

    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.995)
    plt.savefig("results/2x2x2_"+str(data)+".png")
    # plt.show()


    t_stat_1vs0, p_val_1vs0 = stats.ttest_ind(results_3x3_0vs0[data], results_3x3_1vs0[data])
    t_stat_0vs1, p_val_0vs1 = stats.ttest_ind(results_3x3_0vs0[data], results_3x3_0vs1[data])
    t_stat_2vs0, p_val_2vs0 = stats.ttest_ind(results_3x3_0vs0[data], results_3x3_2vs0[data])
    t_stat_0vs2, p_val_0vs2 = stats.ttest_ind(results_3x3_0vs0[data], results_3x3_0vs2[data])
    t_stat_2vs1, p_val_2vs1 = stats.ttest_ind(results_3x3_1vs1[data], results_3x3_2vs1[data])
    t_stat_1vs2, p_val_1vs2 = stats.ttest_ind(results_3x3_1vs1[data], results_3x3_1vs2[data])
    
    fig, ax = plt.subplots(figsize=(26,13))
    boxplot = ax.boxplot([results_3x3_0vs0[data], results_3x3_1vs0[data], results_3x3_0vs1[data], results_3x3_2vs0[data], results_3x3_0vs2[data], results_3x3_1vs1[data], results_3x3_2vs1[data], results_3x3_1vs2[data]],
                            labels=['ToM0 customs vs\n ToM0 smuggler', 'ToM1 customs vs\n ToM0 smuggler', 'ToM0 customs vs\n ToM1 smuggler', 'ToM2 customs vs\n ToM0 smuggler', 
                                    'ToM0 customs vs\n ToM2 smuggler', 'ToM1 customs vs\n ToM1 smuggler', 'ToM2 customs vs\n ToM1 smuggler', 'ToM1 customs vs\n ToM2 smuggler'], patch_artist=True, medianprops={'color': 'black'}
                            )
    colors = ['#A8A9AD', '#728FCE', '#728FCE', '#728FCE', '#728FCE', '#A8A9AD', '#728FCE', '#728FCE']
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)
    plt.ylabel(data, fontsize=30)
    plt.tick_params(axis='y', labelsize=30)
    # plt.title(f"Number of {data} after 365 days in game environment 3")

    height = max([max(results_3x3_0vs0[data]), max(results_3x3_1vs0[data]), max(results_3x3_0vs1[data]), max(results_3x3_2vs0[data]), max(results_3x3_0vs2[data]), max(results_3x3_1vs1[data]), max(results_3x3_2vs1[data]), max(results_3x3_1vs2[data])])
    barplot_annotate_brackets([0,0,0,0,5,5], [1,2,3,4,6,7], [p_val_1vs0, p_val_0vs1, p_val_2vs0, p_val_0vs2, p_val_2vs1, p_val_1vs2], [1,2,3,4,5,6,7,8], [height]*8, dh=[.05, .1, .15, .2, .05, .1])

    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.995)
    plt.savefig("results/3x3_"+str(data)+".png")
    # plt.show()


    t_stat_1vs0, p_val_1vs0 = stats.ttest_ind(results_2x2x2x2_0vs0[data], results_2x2x2x2_1vs0[data])
    t_stat_0vs1, p_val_0vs1 = stats.ttest_ind(results_2x2x2x2_0vs0[data], results_2x2x2x2_0vs1[data])
    t_stat_2vs0, p_val_2vs0 = stats.ttest_ind(results_2x2x2x2_0vs0[data], results_2x2x2x2_2vs0[data])
    t_stat_0vs2, p_val_0vs2 = stats.ttest_ind(results_2x2x2x2_0vs0[data], results_2x2x2x2_0vs2[data])
    t_stat_2vs1, p_val_2vs1 = stats.ttest_ind(results_2x2x2x2_1vs1[data], results_2x2x2x2_2vs1[data])
    t_stat_1vs2, p_val_1vs2 = stats.ttest_ind(results_2x2x2x2_1vs1[data], results_2x2x2x2_1vs2[data])
    
    fig, ax = plt.subplots(figsize=(26,13))
    boxplot = ax.boxplot([results_2x2x2x2_0vs0[data], results_2x2x2x2_1vs0[data], results_2x2x2x2_0vs1[data], results_2x2x2x2_2vs0[data], results_2x2x2x2_0vs2[data], results_2x2x2x2_1vs1[data], results_2x2x2x2_2vs1[data], results_2x2x2x2_1vs2[data]],
                            labels=['ToM0 customs vs\n ToM0 smuggler', 'ToM1 customs vs\n ToM0 smuggler', 'ToM0 customs vs\n ToM1 smuggler', 'ToM2 customs vs\n ToM0 smuggler', 
                                    'ToM0 customs vs\n ToM2 smuggler', 'ToM1 customs vs\n ToM1 smuggler', 'ToM2 customs vs\n ToM1 smuggler', 'ToM1 customs vs\n ToM2 smuggler'], patch_artist=True, medianprops={'color': 'black'}
                            )
    colors = ['#A8A9AD', '#728FCE', '#728FCE', '#728FCE', '#728FCE', '#A8A9AD', '#728FCE', '#728FCE']
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)
    plt.ylabel(data, fontsize=30)
    plt.tick_params(axis='y', labelsize=30)
    # plt.title(f"Number of {data} after 365 days in game environment 2")

    height = max([max(results_2x2x2x2_0vs0[data]), max(results_2x2x2x2_1vs0[data]), max(results_2x2x2x2_0vs1[data]), max(results_2x2x2x2_2vs0[data]), max(results_2x2x2x2_0vs2[data]), max(results_2x2x2x2_1vs1[data]), max(results_2x2x2x2_2vs1[data]), max(results_2x2x2x2_1vs2[data])])
    barplot_annotate_brackets([0,0,0,0,5,5], [1,2,3,4,6,7], [p_val_1vs0, p_val_0vs1, p_val_2vs0, p_val_0vs2, p_val_2vs1, p_val_1vs2], [1,2,3,4,5,6,7,8], [height]*8, dh=[.05, .1, .15, .2, .05, .1])

    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.995)
    plt.savefig("results/2x2x2x2_"+str(data)+".png")
    # plt.show()


    t_stat_1vs0, p_val_1vs0 = stats.ttest_ind(results_4x4_0vs0[data], results_4x4_1vs0[data])
    t_stat_0vs1, p_val_0vs1 = stats.ttest_ind(results_4x4_0vs0[data], results_4x4_0vs1[data])
    t_stat_2vs0, p_val_2vs0 = stats.ttest_ind(results_4x4_0vs0[data], results_4x4_2vs0[data])
    t_stat_0vs2, p_val_0vs2 = stats.ttest_ind(results_4x4_0vs0[data], results_4x4_0vs2[data])
    t_stat_2vs1, p_val_2vs1 = stats.ttest_ind(results_4x4_1vs1[data], results_4x4_2vs1[data])
    t_stat_1vs2, p_val_1vs2 = stats.ttest_ind(results_4x4_1vs1[data], results_4x4_1vs2[data])
    
    fig, ax = plt.subplots(figsize=(26,13))
    boxplot = ax.boxplot([results_4x4_0vs0[data], results_4x4_1vs0[data], results_4x4_0vs1[data], results_4x4_2vs0[data], results_4x4_0vs2[data], results_4x4_1vs1[data], results_4x4_2vs1[data], results_4x4_1vs2[data]],
                            labels=['ToM0 customs vs\n ToM0 smuggler', 'ToM1 customs vs\n ToM0 smuggler', 'ToM0 customs vs\n ToM1 smuggler', 'ToM2 customs vs\n ToM0 smuggler', 
                                    'ToM0 customs vs\n ToM2 smuggler', 'ToM1 customs vs\n ToM1 smuggler', 'ToM2 customs vs\n ToM1 smuggler', 'ToM1 customs vs\n ToM2 smuggler'], patch_artist=True, medianprops={'color': 'black'}
                            )
    colors = ['#A8A9AD', '#728FCE', '#728FCE', '#728FCE', '#728FCE', '#A8A9AD', '#728FCE', '#728FCE']
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)
    plt.ylabel(data, fontsize=30)
    plt.tick_params(axis='y', labelsize=30)
    # plt.title(f"Number of {data} after 365 days in game environment 4")

    height = max([max(results_4x4_0vs0[data]), max(results_4x4_1vs0[data]), max(results_4x4_0vs1[data]), max(results_4x4_2vs0[data]), max(results_4x4_0vs2[data]), max(results_4x4_1vs1[data]), max(results_4x4_2vs1[data]), max(results_4x4_1vs2[data])])
    barplot_annotate_brackets([0,0,0,0,5,5], [1,2,3,4,6,7], [p_val_1vs0, p_val_0vs1, p_val_2vs0, p_val_0vs2, p_val_2vs1, p_val_1vs2], [1,2,3,4,5,6,7,8], [height]*8, dh=[.05, .1, .15, .2, .05, .1])

    plt.tight_layout()
    plt.subplots_adjust(left=0.05, right=0.995)
    plt.savefig("results/4x4_"+str(data)+".png")
    # plt.show()



percentage_increase_2x2_0vs1 = {}; percentage_increase_2x2x2_0vs1 = {}; percentage_increase_3x3_0vs1 = {}; percentage_increase_4x4_0vs1 = {}; percentage_increase_2x2x2x2_0vs1 = {}
percentage_increase_2x2_1vs0 = {}; percentage_increase_2x2x2_1vs0 = {}; percentage_increase_3x3_1vs0 = {}; percentage_increase_4x4_1vs0 = {}; percentage_increase_2x2x2x2_1vs0 = {}
percentage_increase_2x2_1vs2 = {}; percentage_increase_2x2x2_1vs2 = {}; percentage_increase_3x3_1vs2 = {}; percentage_increase_4x4_1vs2 = {}; percentage_increase_2x2x2x2_1vs2 = {}
percentage_increase_2x2_2vs1 = {}; percentage_increase_2x2x2_2vs1 = {}; percentage_increase_3x3_2vs1 = {}; percentage_increase_4x4_2vs1 = {}; percentage_increase_2x2x2x2_2vs1 = {}
for data in ["customs points", "smuggler points", "successful checks", "successful smuggles", "caught packages", "smuggled packages", "total checks", "total smuggles"]:
    percentage_increase_2x2_0vs1[data] = (results_2x2_0vs1[data].values - results_2x2_0vs0[data].values) / abs(results_2x2_0vs1[data].values)
    percentage_increase_2x2x2_0vs1[data] = (results_2x2x2_0vs1[data].values - results_2x2x2_0vs0[data].values) / abs(results_2x2x2_0vs1[data].values)
    percentage_increase_3x3_0vs1[data] = (results_3x3_0vs1[data].values - results_3x3_0vs0[data].values) / abs(results_3x3_0vs1[data].values)
    percentage_increase_2x2x2x2_0vs1[data] = (results_2x2x2x2_0vs1[data].values - results_2x2x2x2_0vs0[data].values) / abs(results_2x2x2x2_0vs1[data].values)
    percentage_increase_4x4_0vs1[data] = (results_4x4_0vs1[data].values - results_4x4_0vs0[data].values) / abs(results_4x4_0vs1[data].values)

    percentage_increase_2x2_1vs0[data] = (results_2x2_1vs0[data].values - results_2x2_0vs0[data].values) / abs(results_2x2_1vs0[data].values)
    percentage_increase_2x2x2_1vs0[data] = (results_2x2x2_1vs0[data].values - results_2x2x2_0vs0[data].values) / abs(results_2x2x2_1vs0[data].values)
    percentage_increase_3x3_1vs0[data] = (results_3x3_1vs0[data].values - results_3x3_0vs0[data].values) / abs(results_3x3_1vs0[data].values)
    percentage_increase_2x2x2x2_1vs0[data] = (results_2x2x2x2_1vs0[data].values - results_2x2x2x2_0vs0[data].values) / abs(results_2x2x2x2_1vs0[data].values)
    percentage_increase_4x4_1vs0[data] = (results_4x4_1vs0[data].values - results_4x4_0vs0[data].values) / abs(results_4x4_1vs0[data].values)

    percentage_increase_2x2_1vs2[data] = (results_2x2_1vs2[data].values - results_2x2_1vs1[data].values) / abs(results_2x2_1vs2[data].values)
    percentage_increase_2x2x2_1vs2[data] = (results_2x2x2_1vs2[data].values - results_2x2x2_1vs1[data].values) / abs(results_2x2x2_1vs2[data].values)
    percentage_increase_3x3_1vs2[data] = (results_3x3_1vs2[data].values - results_3x3_1vs1[data].values) / abs(results_3x3_1vs2[data].values)
    percentage_increase_2x2x2x2_1vs2[data] = (results_2x2x2x2_1vs2[data].values - results_2x2x2x2_1vs1[data].values) / abs(results_2x2x2x2_1vs2[data].values)
    percentage_increase_4x4_1vs2[data] = (results_4x4_1vs2[data].values - results_4x4_1vs1[data].values) / abs(results_4x4_1vs2[data].values)

    percentage_increase_2x2_2vs1[data] = (results_2x2_2vs1[data].values - results_2x2_1vs1[data].values) / abs(results_2x2_2vs1[data].values)
    percentage_increase_2x2x2_2vs1[data] = (results_2x2x2_2vs1[data].values - results_2x2x2_1vs1[data].values) / abs(results_2x2x2_2vs1[data].values)
    percentage_increase_3x3_2vs1[data] = (results_3x3_2vs1[data].values - results_3x3_1vs1[data].values) / abs(results_3x3_2vs1[data].values)
    percentage_increase_2x2x2x2_2vs1[data] = (results_2x2x2x2_2vs1[data].values - results_2x2x2x2_1vs1[data].values) / abs(results_2x2x2x2_2vs1[data].values)
    percentage_increase_4x4_2vs1[data] = (results_4x4_2vs1[data].values - results_4x4_1vs1[data].values) / abs(results_4x4_2vs1[data].values)

plt.rcParams.update({'font.size': 30})

for data in ["customs points", "smuggler points", "successful checks", "successful smuggles", "caught packages", "smuggled packages", "total checks", "total smuggles"]:
    fig, ax = plt.subplots(figsize=(13,13))
    boxplot = ax.boxplot([percentage_increase_2x2_0vs1[data], percentage_increase_3x3_0vs1[data], percentage_increase_4x4_0vs1[data]],
                            labels=[2,3,4], patch_artist=True, medianprops={'color': 'black'}
                            )
    colors = ['#728FCE', '#728FCE', '#728FCE', '#728FCE']
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)

    plt.ylabel(f"Percentage increase of {data}", labelpad=30)
    plt.xlabel("Number of features", labelpad=10)
    # plt.title(f"Percentage increase of {data} after 365 days when ToM0 smuggler becomes ToM1 smuggler against ToM0 customs")

    plt.tight_layout()
    plt.savefig("results/percentageincrease_0vs1_feat_"+str(data)+".png")
    # plt.show()

    fig, ax = plt.subplots(figsize=(13,13))
    boxplot = ax.boxplot([percentage_increase_2x2_0vs1[data], percentage_increase_2x2x2_0vs1[data], percentage_increase_2x2x2x2_0vs1[data]],
                            labels=[2,3,4], patch_artist=True, medianprops={'color': 'black'}
                            )
    colors = ['#728FCE', '#728FCE', '#728FCE', '#728FCE']
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)

    plt.ylabel(f"Percentage increase of {data}", labelpad=30)
    plt.xlabel("Number of categories per feature", labelpad=10)
    # plt.title(f"Percentage increase of {data} after 365 days when ToM0 smuggler becomes ToM1 smuggler against ToM0 customs")

    plt.tight_layout()
    plt.savefig("results/percentageincrease_0vs1_cat_"+str(data)+".png")
    # plt.show()



    fig, ax = plt.subplots(figsize=(13,13))
    boxplot = ax.boxplot([percentage_increase_2x2_1vs0[data], percentage_increase_3x3_1vs0[data], percentage_increase_4x4_1vs0[data]],
                            labels=[2,3,4], patch_artist=True, medianprops={'color': 'black'}
                            )
    colors = ['#728FCE', '#728FCE', '#728FCE', '#728FCE']
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)

    plt.ylabel(f"Percentage increase of {data}", labelpad=30)
    plt.xlabel("Number of features", labelpad=10)
    # plt.title(f"Percentage increase of {data} after 365 days when ToM0 customs becomes ToM1 customs against ToM0 smuggler")

    plt.tight_layout()
    plt.savefig("results/percentageincrease_1vs0_feat_"+str(data)+".png")
    # plt.show()

    fig, ax = plt.subplots(figsize=(13,13))
    boxplot = ax.boxplot([percentage_increase_2x2_1vs0[data], percentage_increase_2x2x2_1vs0[data], percentage_increase_2x2x2x2_1vs0[data]],
                            labels=[2,3,4], patch_artist=True, medianprops={'color': 'black'}
                            )
    colors = ['#728FCE', '#728FCE', '#728FCE', '#728FCE']
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)

    plt.ylabel(f"Percentage increase of {data}", labelpad=30)
    plt.xlabel("Number of categories per feature", labelpad=10)
    # plt.title(f"Percentage increase of {data} after 365 days when ToM0 customs becomes ToM1 customs against ToM0 smuggler")

    plt.tight_layout()
    plt.savefig("results/percentageincrease_1vs0_cat_"+str(data)+".png")
    # plt.show()


    
    fig, ax = plt.subplots(figsize=(13,13))
    boxplot = ax.boxplot([percentage_increase_2x2_1vs2[data], percentage_increase_3x3_1vs2[data], percentage_increase_4x4_1vs2[data]],
                            labels=[2,3,4], patch_artist=True, medianprops={'color': 'black'}
                            )
    colors = ['#728FCE', '#728FCE', '#728FCE', '#728FCE']
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)

    plt.ylabel(f"Percentage increase of {data}", labelpad=30)
    plt.xlabel("Number of features", labelpad=10)
    # plt.title(f"Percentage increase of {data} after 365 days when ToM1 smuggler becomes ToM2 smuggler against ToM1 customs")

    plt.tight_layout()
    plt.savefig("results/percentageincrease_1vs2_feat_"+str(data)+".png")
    # plt.show()

    fig, ax = plt.subplots(figsize=(13,13))
    boxplot = ax.boxplot([percentage_increase_2x2_1vs2[data], percentage_increase_2x2x2_1vs2[data], percentage_increase_2x2x2x2_1vs2[data]],
                            labels=[2,3,4], patch_artist=True, medianprops={'color': 'black'}
                            )
    colors = ['#728FCE', '#728FCE', '#728FCE', '#728FCE']
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)

    plt.ylabel(f"Percentage increase of {data}", labelpad=30)
    plt.xlabel("Number of categories per feature", labelpad=10)
    # plt.title(f"Percentage increase of {data} after 365 days when ToM1 smuggler becomes ToM2 smuggler against ToM1 customs")

    plt.tight_layout()
    plt.savefig("results/percentageincrease_1vs2_cat_"+str(data)+".png")
    # plt.show()



    fig, ax = plt.subplots(figsize=(13,13))
    boxplot = ax.boxplot([percentage_increase_2x2_2vs1[data], percentage_increase_3x3_2vs1[data], percentage_increase_4x4_2vs1[data]],
                            labels=[2,3,4], patch_artist=True, medianprops={'color': 'black'}
                            )
    colors = ['#728FCE', '#728FCE', '#728FCE', '#728FCE']
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)

    plt.ylabel(f"Percentage increase of {data}", labelpad=30)
    plt.xlabel("Number of features", labelpad=10)
    # plt.title(f"Percentage increase of {data} after 365 days when ToM1 customs becomes ToM2 customs against ToM1 smuggler")

    plt.tight_layout()
    plt.savefig("results/percentageincrease_2vs1_feat_"+str(data)+".png")
    # plt.show()

    fig, ax = plt.subplots(figsize=(13,13))
    boxplot = ax.boxplot([percentage_increase_2x2_2vs1[data], percentage_increase_2x2x2_2vs1[data], percentage_increase_2x2x2x2_2vs1[data]],
                            labels=[2,3,4], patch_artist=True, medianprops={'color': 'black'}
                            )
    colors = ['#728FCE', '#728FCE', '#728FCE', '#728FCE']
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)

    plt.ylabel(f"Percentage increase of {data}", labelpad=30)
    plt.xlabel("Number of categories per feature", labelpad=10)
    # plt.title(f"Percentage increase of {data} after 365 days when ToM1 customs becomes ToM2 customs against ToM1 smuggler")

    plt.tight_layout()
    plt.savefig("results/percentageincrease_2vs1_cat_"+str(data)+".png")
    # plt.show()