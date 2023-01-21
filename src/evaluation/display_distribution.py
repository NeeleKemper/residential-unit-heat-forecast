import matplotlib.pyplot as plt
import seaborn as sns
from experiments_eval import get_metric_list

sns.set()


def main():
    # specify the model type: offline or online
    ml = "online"  # online
    # specify the season type of the model: all, summer or winter
    season = "all"  # summer, winter

    offline_experiments = ["cnn", "lr", "dnn"]  # ["lr", "krr", "svr", "rfr", "dnn", "cnn", "lstm"]
    online_experiments = ["odl_er", "odl", "hbp", "rls"]  # ["sgd", odl, hbp, odl_er]

    if ml == "offline":
        experiments = offline_experiments
        metric_key = "rmse"
        xlim_min = 6.4 # 6.4  # 5.6
        xlim_max = 8.3 # 8.3  # 7.3
        binwidth = 0.025
    else:
        experiments = online_experiments
        metric_key = "rmse_24"
        xlim_min = 6.8 # 6.8
        xlim_max = 8.2 # 8.3
        binwidth = 0.01

    metric_list = [get_metric_list(experiment_name=f"{e}_{season}", metric_key=metric_key) for e in experiments]
    x1 = metric_list[0]
    label1 = experiments[0].upper().replace("_", "-")
    x2 = metric_list[1]
    label2 = experiments[1].upper().replace("_", "-")
    x3 = metric_list[2]
    label3 = experiments[2].upper().replace("_", "-")

    plt.figure(figsize=(10, 7), dpi=80)

    kwargs = dict({'alpha': .6})
    sns.histplot(x1, color="dodgerblue", label=label1 + "-Count", binwidth=binwidth, stat="count", multiple="stack", **kwargs)
    sns.histplot(x2, color="orange", label=label2 + "-Count", binwidth=binwidth, stat="count",  multiple="stack", **kwargs)
    sns.histplot(x3, color="deeppink", label=label3 + "-Count", binwidth=binwidth, stat="count", multiple="stack", **kwargs)
    if ml == "online":
        x4 = metric_list[3]
        label4 = experiments[3].upper().replace("_", "-")
        plt.axvline(x=x4[0], color='teal', label=label4, linestyle="--", linewidth="2")

    if ml == "offline":
        kwargs = dict({'linewidth': 2})
        sns.kdeplot(x1, color="dodgerblue", label=label1 + "-KDE", **kwargs)
        sns.kdeplot(x2, color="orange", label=label2 + "-KDE", **kwargs)
        sns.kdeplot(x3, color="deeppink", label=label3 + "-KDE", **kwargs)


    # kwargs = dict(hist_kws={'alpha': .6}, kde_kws={'linewidth': 2})
    # sns.distplot(x1, color="dodgerblue", label=label1, bins=10, hist=True, kde=True, **kwargs)
    # sns.distplot(x2, color="orange", label=label2, bins=10, hist=True, kde=True, **kwargs)
    # sns.distplot(x3, color="deeppink", label=label3, bins=10, hist=True, kde=True, **kwargs)

    plt.grid(True, which="both")
    plt.xlim(xlim_min, xlim_max)
    fontsize = 16
    plt.xlabel("RMSE [kWh]", fontsize=fontsize)
    plt.ylabel("Numerosity in a bin", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(prop={'size': fontsize}, ) #loc='upper center', bbox_to_anchor=(0.4, 0.5, 0.5, 0.5)
    plt.tight_layout()

    if ml == "offline":
        plt.savefig(f'distribution_plots/{label1}_{label2}_{label3}.png')
    else:
        plt.savefig(f'distribution_plots/{label1}_{label2}_{label3}_{label4}.png')
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
