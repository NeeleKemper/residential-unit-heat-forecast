import datetime
import pytz
import glob
import warnings
import pandas as pd
warnings.simplefilter(action="ignore", category=FutureWarning)


class DBProcessing:
    def __init__(self):
        self.df_heat = pd.DataFrame()
        self.root_path = "data/database"
        self.filename = "db_data"

    @staticmethod
    def _utc_to_local(utc_time: str) -> datetime:
        """

        :param utc_time:
        :return:
        """
        local_timezone = pytz.timezone("Europe/Berlin")

        utc_time = datetime.datetime.strptime(utc_time, "%Y-%m-%d %H:%M:%S")
        local_datetime = utc_time.replace(
            tzinfo=pytz.utc).astimezone(local_timezone)
        return local_datetime.replace(tzinfo=None)

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """

        :param df:
        :return:
        """
        columns = dict()
        with open(f"{self.root_path}/sensor_id_mapping.txt") as f:
            lines = f.readlines()
            for line in lines:
                values = line.split("-")
                key = values[1].replace(" ", "").rstrip()
                val = values[0].rstrip()
                columns[key] = val
        df = df.rename(columns=columns)
        return df

    def process_data(self):
        """

        :return:
        """
        files = glob.glob(f"{self.root_path}/*.csv")
        for f in files:
            df_temp = pd.read_csv(f, sep=";")
            self.df_heat = self.df_heat.append(df_temp)
        self.df_heat = self._index_to_dt(self.df_heat)
        self.df_heat = self._rename_columns(self.df_heat)
        # scale data up to 5 minutes
        self.df_heat = self.df_heat.astype(float).resample("300s").mean()
        # kilowatt into watt
        self.df_heat = self.df_heat*1000

    def _index_to_dt(self, df: pd.DataFrame) -> pd.DataFrame:
        """

        :param df:
        :return:
        """
        unix_time = df["Unnamed: 0"].to_list()
        local_dt = list()
        for u in unix_time:
            gmt_dt = datetime.datetime.utcfromtimestamp(u).strftime("%Y-%m-%d %H:%M:%S")
            dt = self._utc_to_local(str(gmt_dt))
            local_dt.append(dt)
        df.drop(["Unnamed: 0"], inplace=True, axis=1)
        df["datetime"] = local_dt
        df.set_index("datetime", drop=True, inplace=True)
        df = df.sort_index()
        return df

    def save_data(self) -> None:
        """

        :return:
        """
        self.df_heat.index.name = "datetime"
        self.df_heat.to_csv(f"data/{self.filename}.csv", sep=";")

    def __call__(self) -> pd.DataFrame:
        self.process_data()
        self.save_data()
        return self.df_heat
