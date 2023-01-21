import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

sns.set()


def load_dataframe() -> pd.DataFrame:
    """

    :return:
    """
    df = pd.read_csv("data/processed_data.csv", sep=";", index_col="datetime")
    df.index = pd.to_datetime(df.index)
    return df


def plot_heat_curve(df: pd.DataFrame, save_plot: bool) -> None:
    """

    :param df:
    :param save_plot:
    :return:
    """
    plots = [
        {"plot": "heat_scaled", "y_label": "Heat [kW]", "title": "Scaled Heat", "scaling": 0.001},
        {"plot": "avg_24", "y_label": "Heat [kW]", "title": "Average Scaled Heat of the last 24 Hours",
         "scaling": 0.001},
        {"plot": "last_24", "y_label": "Heat [kW]", "title": "Scaled Heat 24 Hours ago", "scaling": 0.001},
        {"plot": "temp", "y_label": "Temperature [°C]", "title": "Temperature", "scaling": 1},
        {"plot": "degree_hour", "y_label": "Temperature [°C]", "title": "Degree Hour", "scaling": 1},
        {"plot": "solar_rad", "y_label": "Solar radiation [W/m²]", "title": "Solar Radiation", "scaling": 1},
    ]
    for p in plots:
        name = p["plot"]
        y_label = p["y_label"]
        x_label = "Time [h]"
        color = "darkslateblue"
        value = df[name]
        x_time = df.index.tolist()
        value = value * p["scaling"]
        fig, ax = plt.subplots()
        fontsize=10
        ax.plot(x_time, value, linewidth=0.5, color=color)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.set_ylabel(y_label, fontsize=fontsize)

        # format the ticks
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m\n%Y"))

        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%m"))

        date_min = np.datetime64(x_time[0], "D")
        date_max = np.datetime64(x_time[-1], "D")

        ax.set_xlim(date_min, date_max)

        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize)

        ax.grid(True, which="both")
        #plt.title(p["title"])
        plt.tight_layout()

        if save_plot:
            plt.savefig(f"data_plots/{name}.png")

        plt.show()
        plt.close()


def define_heating_season(df: pd.DataFrame, section: str = "year") -> None:
    """
    examine the heating data for the beginning and end of the heating season. For this purpose, the heating data of the
    years 2020 and 2021 are superimposed and the limits are determined.
    :param df:
    :param section: Specifies whether the beginning (September) of the heating season should be examined, or the end
        (May). By default, the superimposed years are plotted.
    :return:
    """
    y_label = "Heat [kW]"
    x_label = "Time"
    split = "2021-06-01 00:00:00"
    title = "Year"
    vs = 0

    index_dt = [str(dt) for dt in df.index.tolist()]
    index_split = index_dt.index(split)
    df_1 = df.iloc[:index_split - 1, :]
    df_2 = df.iloc[index_split:, :]

    x_time = df_1.index.tolist()
    value1 = df_1["heat_scaled"].to_list()

    value2 = df_2["heat_scaled"].to_list()
    width = len(value1) - len(value2)
    value2.extend([np.nan] * width)

    sum_list = []
    for (item1, item2) in zip(value1, value2):
        sum_list.append(item1 + item2)

    if section == "begin":
        n_samples = 2208
        x_time = x_time[n_samples:n_samples + 720]
        value1 = value1[n_samples:n_samples + 720]
        value2 = value2[n_samples:n_samples + 720]
        sum_list = sum_list[n_samples:n_samples + 720]
        title = "September"
        vs = np.datetime64("2020-09-21")

    if section == "end":
        title = "May"
        n_samples = 743
        value1 = value1[-n_samples:]
        x_time = x_time[-n_samples:]
        value2 = [[np.nan] * n_samples][0]
        sum_list = [[np.nan] * n_samples][0]
        vs = np.datetime64("2021-05-09")

    fig, ax = plt.subplots()
    ax.plot(x_time, value1, linewidth=0.5, color="b", label="2020")
    ax.plot(x_time, value2, linewidth=0.5, color="g", label="2021")
    ax.plot(x_time, sum_list, linewidth=0.5, color="r", label="sum")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if section != "year":
        # format the ticks
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
        plt.xticks(rotation=90)
        plt.axvline(x=vs, color="black", ls=":")
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    date_min = np.datetime64(x_time[0], "D")
    date_max = np.datetime64(x_time[-1], "D")

    ax.set_xlim(date_min, date_max)

    ax.grid(True, which="minor")
    plt.tight_layout()
    plt.legend()
    plt.title(title)

    plt.show()
    plt.close()


def main():
    df = load_dataframe()
    plot_heat_curve(df, save_plot=False)
    # define_heating_season(df, section="begin")


if __name__ == "__main__":
    # execute only if run as a script
    main()
