import glob
import datetime
import re
import warnings
import pandas as pd
from typing import Tuple

warnings.simplefilter(action="ignore", category=FutureWarning)


class ControllerProcessing:
    def __init__(self):
        self.files_path = ["apartments_01_22", "apartments_23_44", "apartments_45_66"]
        self.root_path = "data/controller"
        self.filename = "controller_data"
        columns, idx = self._create_index()
        self.df_heat = pd.DataFrame(columns=columns, index=idx)

    @staticmethod
    def _create_index() -> Tuple[list, pd.Index]:
        """

        :return:
        """
        idx = pd.date_range(start="06.01.2020 00:05:00", end="03.01.2022 00:00:00", freq="300s")
        columns = [f"Heat:Wohnung {i}" for i in range(1, 67)]
        idx = idx.strftime("%Y-%m-%d %H:%M:%S")
        return columns, idx

    @staticmethod
    def _date_parser(x: str) -> datetime:
        """

        :param x:
        :return:
        """
        try:
            d = datetime.datetime.strptime(str(x), "%d.%m.%Y %H:%M:%S")
        except(Exception,):
            d = datetime.datetime.strptime(str(x), "%d.%m.%Y %H:%M")
        return d

    @staticmethod
    def _round_minutes(dt: datetime, resolution: int) -> str:
        """

        :param dt:
        :param direction:
        :param resolution:
        :return:
        """
        new_minute = (dt.minute // resolution + 1) * resolution
        new_dt = dt + datetime.timedelta(minutes=new_minute - dt.minute)
        new_dt = new_dt.replace(second=0)
        return new_dt.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _filter_columns(df: pd.DataFrame) -> pd.DataFrame:
        """

        :param df:
        :return:
        """
        columns = df.columns.values
        r = re.compile(r"((Heat:Wohnung\s)(\d{1,2})$)|datetime")
        col = list(filter(r.match, columns))
        return df[col]

    def concatenation_of_apartments(self) -> None:
        """

        :return:
        """
        for path in self.files_path:
            # get direction of csv files for a group of apartments
            zl_files = glob.glob(f"{self.root_path}/{path}/zl_*.csv")
            # connect all apartments of a block
            df_temp = self._concatenation_of_zl_csv(zl_files)
            # set index to datetime
            df_temp.set_index("datetime", inplace=True)

            df_temp = df_temp[~df_temp.index.duplicated(keep="first")]
            self.df_heat = self.df_heat[~self.df_heat.index.duplicated(keep="first")]
            self.df_heat.update(df_temp)

    def _concatenation_of_zl_csv(self, files: list) -> pd.DataFrame:
        """

        :param files:
        :return:
        """
        df = pd.DataFrame()
        for f in files:
            df_temp = pd.read_csv(f, sep=";")
            if "tkt=300,Leistung_in_W" in df_temp.columns:
                # drop rows where time index is Nan
                df_temp = df_temp[df_temp["tkt=300,Leistung_in_W"].notna()]
                # convert time to datetime
                df_temp["tkt=300,Leistung_in_W"] = df_temp["tkt=300,Leistung_in_W"].apply(
                    lambda x: self._date_parser(x))
                # rename column with time specification
                df_temp.rename(columns={"tkt=300,Leistung_in_W": "datetime"}, inplace=True)
                df_temp["datetime"] = df_temp["datetime"].apply(
                    lambda x: self._round_minutes(x, 5))
            else:
                continue
            # df.concat(df_temp)
            df = df.append(df_temp)

        # sort dataframe by time
        df.sort_values(by="datetime", inplace=True)

        # filter needed columns
        df = self._filter_columns(df)

        return df

    def save_data(self) -> None:
        """

        :return:
        """
        self.df_heat.index.name = "datetime"
        self.df_heat.to_csv(f"data/{self.filename}.csv", sep=";")

    def __call__(self) -> pd.DataFrame:
        self.concatenation_of_apartments()
        self.save_data()
        return self.df_heat
