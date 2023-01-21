import pandas as pd

from src.time_series.utils import load_time_series
from src.utils.contant import SUMMER_BEGIN, SUMMER_END, WINTER_BEGIN, WINTER_END


class DataPipelineTS:
    def __init__(self, season: str):
        assert season in ["all", "summer", "winter"], "The season parameter is not defined. It must be 'all', 'winter' " \
                                                      "or 'summer'"
        self.season = season
        self.train_index = None
        self.test_index = None
        self.train_data = None
        self.test_data = None
        self.df = load_time_series()

    def split_data(self):
        if self.season == "all":
            split = "2021-06-01 00:00:00"
            index_dt = [str(dt) for dt in self.df.index.tolist()]
            index_split = index_dt.index(split)
            df_train = self.df.iloc[:index_split - 1]
            df_test = self.df.iloc[index_split:]
        else:
            if self.season == "summer":
                dt_train = pd.date_range(start=SUMMER_BEGIN[0], end=SUMMER_END[0], freq='H')
                dt_test = pd.date_range(start=SUMMER_BEGIN[1], end=SUMMER_END[1], freq='H')
            else:
                # self.season == "winter":
                dt_train = pd.date_range(start=WINTER_BEGIN[0], end=WINTER_END[0], freq='H')
                dt_test = pd.date_range(start=WINTER_BEGIN[1], end=WINTER_END[1], freq='H')

            train_index = [x for x in self.df.index if x in dt_train]
            test_index = [x for x in self.df.index if x in dt_test]
            df_train = self.df.loc[train_index]
            df_test = self.df.loc[test_index]

        self.train_index = df_train.index
        self.test_index = df_test.index
        self.train_data = df_train.values
        self.test_data = df_test.values

    def __call__(self):
        self.split_data()
        return self.train_data, self.test_data
