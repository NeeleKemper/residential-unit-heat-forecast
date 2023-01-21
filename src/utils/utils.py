import argparse
import pandas as pd
import numpy as np
from typing import Tuple
from src.utils.contant import SUMMER_BEGIN, SUMMER_END, WINTER_BEGIN, WINTER_END

def parse_season() -> str:
    """

    :return:
    """
    # Create the parser
    parser = argparse.ArgumentParser(description="Season input for model")
    # Add the arguments
    parser.add_argument("Season",
                        metavar="season",
                        type=str,
                        help="season can be 'summer', 'winter' or 'all'.")
    # Execute the parse_args() method
    args = parser.parse_args()
    season = args.Season

    assert season in ["all", "summer", "winter"], "The season parameter is not defined. It must be 'all', 'winter' " \
                                                  "or 'summer'"
    return season


def load_data() -> pd.DataFrame:
    """

    :return:
    """
    df = pd.read_csv("../data_processing/data/processed_data.csv", sep=";", index_col="datetime")
    df.index = pd.to_datetime(df.index)
    df.dropna(axis=0, inplace=True)
    return df


def split_season(season: str, df: pd.DataFrame) -> pd.DataFrame:
    """

    :return:
    """
    assert season != "all", "No season split necessary for season 'all'!"
    if season == "summer":
        dt_a = pd.date_range(start=SUMMER_BEGIN[0], end=SUMMER_END[0], freq='H')
        dt_b = pd.date_range(start=SUMMER_BEGIN[1], end=SUMMER_END[1], freq='H')
    else:
        dt_a = pd.date_range(start=WINTER_BEGIN[0], end=WINTER_END[0], freq='H')
        dt_b = pd.date_range(start=WINTER_BEGIN[1], end=WINTER_END[1], freq='H')
    dt = np.concatenate((dt_a, dt_b), axis=None)
    index = [x for x in df.index if x in dt]
    return df.loc[index]


def watt_into_kilowatt(values: np.ndarray) -> np.ndarray:
    """

    :param values:
    :return:
    """
    return np.round(values/1000, 2)