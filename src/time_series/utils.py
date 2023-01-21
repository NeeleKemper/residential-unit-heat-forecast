import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import seaborn as sns
from src.utils.utils import load_data
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

sns.set()


def load_time_series() -> pd.DataFrame:
    """

    :return:
    """
    df = load_data()
    df = df["heat_scaled"]
    return df


def analyse_seasonal_decompose(df) -> None:
    """

    :param df:
    :return:
    """
    result = seasonal_decompose(df, model="additive", period=24)
    trend = result.trend
    seasonal = result.seasonal
    resid = result.resid
    observed = result.observed
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)

    # plot observed

    d = df.index
    date_min = np.datetime64(d[0], "D")
    date_max = np.datetime64(d[-1], "D")

    ax1.plot(df.index, observed, linewidth=0.1)
    ax1.set_ylabel("Observed")
    ax1.set_xlabel("")
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d.\n%b"))
    ax1.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax1.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax1.set_xlim(date_min, date_max)

    # plot trend
    ax2.plot(d, trend, linewidth=0.1)
    ax2.set_ylabel("Trend")
    ax2.set_xlabel("")
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d.\n%b"))
    ax2.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax2.xaxis.set_minor_formatter(mticker.NullFormatter())

    # plot seasonal
    ax3.plot(d, seasonal, linewidth=0.1)
    ax3.set_ylabel("Season")
    ax3.set_xlabel("")

    ax3.xaxis.set_major_locator(mdates.MonthLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%d.\n%b"))
    ax3.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax3.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax3.set_xlim(date_min, date_max)

    # plot resid
    ax4.plot(d, resid, linewidth=0.1)
    ax4.set_ylabel("Resid")
    ax4.set_xlabel("Time")

    ax4.xaxis.set_major_locator(mdates.MonthLocator())
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%d.\n%b"))
    ax4.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    ax4.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax4.set_xlim(date_min, date_max)
    fig.suptitle("Seasonal decomposition")

    plt.tight_layout()
    plt.show()


def check_stationarity(df) -> None:
    """

    :param df:
    :return:
    """
    result = adfuller(df, autolag="AIC")
    output = pd.Series(result[0:4],
                       index=["Test Statistic", "p-value", "#Lags Used", "Number of Observations Used"])
    print(output)
    print("Critical Values:")
    for key, value in result[4].items():
        print("%s: %.3f" % (key, value))


def main():
    df = load_time_series()
    analyse_seasonal_decompose(df)
    check_stationarity(df)


if __name__ == "__main__":
    # execute only if run as a script
    main()
