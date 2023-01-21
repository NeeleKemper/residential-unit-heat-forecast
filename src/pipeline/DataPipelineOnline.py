import warnings
import pandas as pd
from src.utils.utils import load_data, split_season

warnings.simplefilter(action="ignore", category=FutureWarning)


class DataPipelineOnline:

    def __init__(self, season: str):
        assert season in ["all", "summer", "winter"], "The season parameter is not defined. It must be 'all', 'winter' " \
                                                      "or 'summer'"
        self.season = season
        self.df = load_data()

    def split_by_days(self) -> list:
        """

        :return:
        """
        self.df.index = pd.to_datetime(self.df.index.tolist())
        days = [g for n, g in self.df.groupby(pd.Grouper(freq="1D"))]
        days = [df for df in days if not df.empty]
        return days

    def data_generator(self, days: list) -> None:
        """

        :param days:
        :return:
        """
        for i in range(2, len(days)):
            d_train = days[i - 2]
            d_test = days[i - 1]

            d_test = d_test.append(days[i])

            y_train = d_train['heat_scaled'].to_frame()
            df_temp = d_train.drop(["heat_scaled"], axis=1)
            X_train = df_temp.values

            y_test = d_test['heat_scaled'].to_frame()
            df_temp = d_test.drop(["heat_scaled"], axis=1)

            X_test = df_temp.values

            yield X_train, X_test, y_train.to_numpy(), y_test.to_numpy(), d_test.index

    def __call__(self):
        if self.season != "all":
            self.df = split_season(season=self.season, df=self.df)
        days = self.split_by_days()
        return self.data_generator(days)
