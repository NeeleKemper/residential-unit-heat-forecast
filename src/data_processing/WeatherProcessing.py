import datetime
import pytz
import glob
import pandas as pd


class WeatherProcessing:
    def __init__(self):
        self.df_weather = pd.DataFrame()
        self.root_path = "data/weather"
        self.filename = "weather_data"

    @staticmethod
    def _utc_to_local(utc_time: str) -> datetime:
        """

        :param utc_time:
        :return:
        """
        local_timezone = pytz.timezone("Europe/Berlin")
        try:
            utc_time = datetime.datetime.strptime(utc_time, "%Y-%m-%d %H:%M:%S")
        except(Exception,):
            utc_time = datetime.datetime.strptime(utc_time, "%d.%m.%Y %H:%M")
        local_datetime = utc_time.replace(
            tzinfo=pytz.utc).astimezone(local_timezone)
        return local_datetime.replace(tzinfo=None)

    def _convert_index(self, df):
        """

        :param df:
        :return:
        """
        # convert utc time to local time

        df["datetime"] = df["datetime"].apply(lambda x: self._utc_to_local(x))
        df = df[["datetime", "temp", "solar_rad"]]
        df.set_index("datetime", drop=True, inplace=True)
        # drop duplicated index
        df = df[~df.index.duplicated(keep="first")]
        # time change
        df_temp = pd.DataFrame(data=[df.loc["2021-03-25 03:00:00"].tolist()], columns=df.columns.tolist(),
                               index=["2021-03-25 02:00:00"])
        df = df.append(df_temp)
        # convert index to datetime object
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df

    def process_data(self):
        """

        :return:
        """
        files = glob.glob(f"{self.root_path}/*.csv")
        for f in files:
            df_temp = pd.read_csv(f, sep=";")
            self.df_weather = self.df_weather.append(df_temp)
        self.df_weather = self._convert_index(self.df_weather)
        self.df_weather = self.df_weather.dropna()
        self.df_weather["temp"] = self.df_weather["temp"].round()
        self.df_weather = self.df_weather.drop(self.df_weather[self.df_weather.index < "2020-06-01 00:00:00"].index)
        self.df_weather = self.df_weather.drop(self.df_weather[self.df_weather.index > "2022-03-01 00:00:00"].index)

    def save_data(self) -> None:
        """

        :return:
        """
        self.df_weather.index.name = "datetime"
        self.df_weather.to_csv(f"data/{self.filename}.csv", sep=";")

    def __call__(self) -> pd.DataFrame:
        self.process_data()
        self.save_data()
        return self.df_weather