import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import seaborn as sns

sns.set()

x = pd.date_range(start="2021-11-29 00:00:00", end="2021-12-01 23:00:00", freq="H")
y_true = []
online_experiments = ["sgd", "rls", "odl", "odl_er", "hbp"]
color_list = ["purple","teal", "orange", "dodgerblue", "deeppink"]
linestyle= ["-", "--", "--", "-.", ":"]
fontsize = 10

fig, ax = plt.subplots()
for model, color, ls in zip(online_experiments,color_list, linestyle):
    df = pd.read_csv(f"eval_over_time/{model}_all_prediction.csv", sep=";", index_col="date")
    df.index = pd.to_datetime(df.index)
    df = df.loc[x]
    if len(y_true) == 0:
        y_true = df["heat"]
        ax.plot(x, y_true, label="Heat", color="darkgray")
    label = model.replace("_", "-").upper()
    ax.plot(x, df["pred"], label=label, color=color, linestyle=ls, linewidth='2')

ax.set_xlabel("Time [h]", fontsize=fontsize)
ax.set_ylabel("Heat [kW]",  fontsize=fontsize)

ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%Y"))

date_min = np.datetime64(x[0])
date_max = np.datetime64(x[-1])

ax.set_xlim(date_min, date_max)
ax.grid(True, which="both")
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(fontsize=fontsize+1)
plt.tight_layout()
plt.savefig("eval_over_time/pred_comparison.png")
plt.show()
plt.close()


