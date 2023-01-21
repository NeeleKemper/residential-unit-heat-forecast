import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.utils.utils import load_data, split_season


class DataPipelineOffline:
    def __init__(self, season: str):
        assert season in ["all", "summer", "winter"], "The season parameter is not defined. It must be 'all', 'winter' " \
                                                      "or 'summer'"
        self.season = season
        self.X = None
        self.X_summer = None
        self.X_winter = None
        self.y = None
        self.index = None
        self.df = load_data()
        self.scaler = MinMaxScaler()

    def normalize(self, X: np.ndarray) -> np.ndarray:
        """
        normalize the X values between 0 and 1
        :param X:
        :return:
        """
        X_norm = self.scaler.fit_transform(X)
        return X_norm

    def _split_data(self, df: pd.DataFrame) -> None:
        """
        split normalized data into summer and winter data
        :param df: data frame with normalized data
        :return:
        """
        columns = df.columns
        index = df.index
        df_norm = pd.DataFrame(data=self.X, columns=columns, index=index)
        df_summer = split_season(season="summer", df=df_norm)
        df_winter = split_season(season="winter", df=df_norm)
        self.X_summer = df_summer.values
        self.X_winter = df_winter.values

    def process_data(self) -> None:
        """

        :return:
        """
        self.index = self.df.index.tolist()
        self.y = np.array(self.df["heat_scaled"].tolist())
        df_temp = self.df.drop(["heat_scaled"], axis=1)
        X = df_temp.values
        self.X = self.normalize(X)

        if self.season == "all":
            self._split_data(df_temp)

    def __call__(self):
        if self.season != "all":
            self.df = split_season(season=self.season, df=self.df)
        self.process_data()
        return self.X, self.y
