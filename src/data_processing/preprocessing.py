import warnings
import datetime
import pandas as pd
import numpy as np
from os import path
from typing import Tuple

from ContollerProcessing import ControllerProcessing
from DBProcessing import DBProcessing
from WeatherProcessing import WeatherProcessing

warnings.simplefilter(action="ignore", category=FutureWarning)


def load_dataframes() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """

    :return:
    """
    root_path = "data"
    if path.exists(f"{root_path}/db_data.csv"):
        df_db = pd.read_csv(f"{root_path}/db_data.csv", sep=";", index_col="datetime")
        df_db.index = pd.to_datetime(df_db.index)
    else:
        dp = DBProcessing()
        df_db = dp()

    if path.exists(f"{root_path}/controller_data.csv"):
        df_c = pd.read_csv(f"{root_path}/controller_data.csv", sep=";", index_col="datetime")
        df_c.index = pd.to_datetime(df_c.index)
    else:
        cp = ControllerProcessing()
        df_c = cp()

    if path.exists(f"{root_path}/weather_data.csv"):
        df_w = pd.read_csv(f"{root_path}/weather_data.csv", sep=";", index_col="datetime")
        df_w.index = pd.to_datetime(df_w.index)
    else:
        wp = WeatherProcessing()
        df_w = wp()
    return df_db, df_c, df_w


def merge_dataframes(df_db: pd.DataFrame, df_c: pd.DataFrame) -> pd.DataFrame:
    """

    :param df_db:
    :param df_c:
    :return:
    """
    df_db[df_db.isnull()] = df_c
    df_db = df_db.drop(df_db[df_db.index < "2020-08-01 00:00:00"].index)
    df_begin = df_c[df_c.index < "2020-08-01 00:00:00"]
    df = df_db.append(df_begin)
    df = df.sort_index()
    return df


def delete_frozen_heat_counter(df: pd.DataFrame) -> pd.DataFrame:
    """

    :param df:
    :return:
    """
    date_frozen = pd.date_range(
        start="06.09.2020 01:00:00", end="06.09.2020 16:00:00", freq="H")
    df = df.drop(date_frozen)

    date_frozen = pd.date_range(
        start="12.20.2020 09:00:00", end="12.20.2020 23:00:00", freq="H")
    df = df.drop(date_frozen)

    date_frozen = pd.date_range(
        start="12.21.2020 00:00:00", end="12.21.2020 17:00:00", freq="H")
    df = df.drop(date_frozen)
    return df


def average_hourly_value(df: pd.DataFrame) -> pd.DataFrame:
    """

    :param df:
    :return:
    """
    df.index = pd.to_datetime(df.index.tolist())
    # calculate mean value over one hour
    df = df.astype(float).resample("1H").mean()
    # round the mean
    return df


def calculate_mean(df: pd.DataFrame) -> pd.DataFrame:
    """

    :param df:
    :return:
    """
    columns = df.columns.values
    heat_columns = [c for c in columns if "Heat" in c]
    # mean over all apartments
    df["heat_mean"] = df[heat_columns].mean(skipna=True, axis=1)
    return df


def calculate_scaled(df: pd.DataFrame) -> pd.DataFrame:
    """

    :param df:
    :return:
    """
    df["heat_scaled"] = df["heat_mean"].apply(lambda x: round(x * 66, 2))
    return df


def _get_avg_24(df: pd.DataFrame) -> pd.Series:
    """

    :param df:
    :return:
    """
    s = df["heat_scaled"].rolling("24H").mean()
    return s


def _get_last_24(df: pd.DataFrame, input_time: datetime) -> pd.DataFrame:
    time = input_time - datetime.timedelta(hours=24)
    try:
        last_heat = df.loc[time]["heat_scaled"]
    except (Exception,):
        # if there was no data before 24 hours.
        last_heat = df.loc[input_time]["heat_scaled"]
    return last_heat


def _get_weekday(time: datetime) -> int:
    """

    :param time:
    :return:
    """
    holiydays = ["1.6.2020", "11.6.2020", "8.8.2020", "15.8.2020", "3.10.2020", "1.11.2020 ", "25.12.2020",
                 "26.12.2020", "1.1.2021", "1.6.2021", "2.4.2021", "5.4.2021", "1.5.2021", "13.5.2021", "24.5.2021",
                 "3.6.2021", "15.8.2021", "3.10.2021", "1.11.2021", "25.12.2021", "26.12.2021", "1.1.2022", "6.1.2022"]
    date = f"{time.day}.{time.month}.{time.year}"
    day = time.dayofweek
    # weekend
    if day == 5 or day == 6 or date in holiydays:
        return 1
    # weekday
    return 0


def _degree_hour(series: pd.Series) -> float:
    """

    :param series:
    :return:
    """
    heating_limit = 15.0
    sol_temp = 21.0
    if series.avg_day_temp < heating_limit:
        return max(round(sol_temp - series.temp, 2), 0)
    return 0


def degree_hour_number(df: pd.DataFrame) -> pd.DataFrame:
    """

    :param df:
    :return:
    """
    df_avg = df["temp"].groupby(pd.Grouper(freq="d")).mean().values
    df_avg = list(np.repeat(df_avg, 24))

    idx = pd.date_range(
        start="06.01.2020 00:00:00", end="03.01.2022 00:00:00", freq="H")
    df = df.reindex(idx, fill_value=np.nan)

    df["avg_day_temp"] = df_avg[:-23]
    df["degree_hour"] = df[["temp", "avg_day_temp"]].apply(_degree_hour, axis=1)
    df = df.drop(["avg_day_temp"], axis=1)
    df = df.dropna(axis=0)
    return df


def calculate_time_data(df: pd.DataFrame) -> pd.DataFrame:
    """

    :param df:
    :return:
    """
    # month of year
    month_of_year_series = df.index.to_series().apply(lambda x: x.month)
    df.insert(0, "moy", value=month_of_year_series)
    # day of week
    day_of_week_series = df.index.to_series().apply(lambda x: x.dayofweek)
    df.insert(1, "dow", value=day_of_week_series)
    # hour of day
    hour_of_day_series = df.index.to_series().apply(lambda x: x.hour)
    df.insert(2, "hod", value=hour_of_day_series)
    # get the average heat consumption over the last 24 hours
    avg_24 = _get_avg_24(df)
    df.insert(4, "avg_24", value=avg_24)
    # get the heat consumption 24 hours again
    last_24 = df.index.to_series().apply(lambda x: _get_last_24(df, x))
    last_24 = last_24.fillna(0)
    df.insert(5, "last_24", value=last_24)
    return df


def save_data(df: pd.DataFrame) -> None:
    df.index.name = "datetime"
    df.to_csv(f"data/processed_data.csv", sep=";")


def main():
    df_db, df_c, df_w = load_dataframes()
    df = merge_dataframes(df_db, df_c)
    df = calculate_mean(df)
    df = average_hourly_value(df)
    df = calculate_scaled(df)
    df_heat = df["heat_scaled"]

    # adding temp and solar data
    df_heat = pd.concat([df_heat, df_w], axis=1)
    df_heat = delete_frozen_heat_counter(df_heat)

    df_heat = calculate_time_data(df_heat)
    df_heat = degree_hour_number(df_heat)
    df_heat = df_heat.dropna()
    # dropping last row
    df_heat.drop(df_heat.tail(1).index, inplace=True)
    save_data(df_heat)


if __name__ == "__main__":
    # execute only if run as a script
    main()