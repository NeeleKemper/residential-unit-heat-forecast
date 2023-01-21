import datetime
import time
import pandas as pd
import mysql.connector
from mysql.connector import errorcode

from db_config import USER, PASSWORD, HOST, DB


def create_date_list() -> list:
    """
    create a list of tuples containing the start, end times and filename for retrieving and storing the heating data
    from the database
    :return: list of tuples [(start time: start of a day, end time: end of the same day, filename: date of the day)]
    """
    # list of dates between 1.june 2020 up to and including 25.february 2021
    dates = pd.date_range(start="07.01.2020 00:00:00", end="02.25.2022 23:00:00", freq="D")
    date_list = list()
    for d in dates:
        # start of the day in unix time
        start_date = int(time.mktime(d.timetuple()))
        added_seconds = datetime.timedelta(0, 86400 - 1)
        new_datetime = d + added_seconds
        # end of the day in unix time
        end_date = int(time.mktime(new_datetime.timetuple()))
        #  date of the day
        filename = d.strftime("%Y_%m_%d")
        date_list.append((start_date, end_date, filename))
    return date_list


class SQLDatabase:

    def __init__(self):
        self.cnx = None
        self.cursor = None
        self.sensor_ids = list()
        self.df = pd.DataFrame()

    def establish_connection(self) -> None:
        """
        establish a connection to the database
        :return:
        """
        try:
            self.cnx = mysql.connector.connect(user=USER,
                                               password=PASSWORD,
                                               host=HOST,
                                               database=DB)
            print(f"Connection established: {self.cnx}")
            return self.cnx

        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("Something is wrong with your user name or password")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("Database does not exist")
            else:
                print(err)

    def get_sensor_ids(self) -> None:
        """
        retrieves the sensor ids for the heat meters of the apartments
        :return:
        """
        # sql request
        sql_select_query = "select * from T_Sensors"
        self.cursor = self.cnx.cursor()
        self.cursor.execute(sql_select_query)
        records = self.cursor.fetchall()
        # extract sensor ids
        for row in records:
            if "PowerHeat" in row[2] and row[0] != 812656082 and row[0] != 812656055:
                print(f"{row[7]} - Sensor id: {row[0]}")
                self.sensor_ids.append(row[0])
        # set the sensor ids as column indexes for dataframe.
        self.df = pd.DataFrame(columns=self.sensor_ids)

    def get_history(self, start_ts: int, end_ts: int) -> None:
        """
        retrieves the historical power data of the heat meters of the apartments between start date and end date.
        :param start_ts: start of the period as unix timestamp
        :param end_ts: end of the period as unix timestamp
        :return:
        """
        ids = tuple(self.sensor_ids)
        # sql request
        sql_select_query = f"select * from T_History where T_Sensors_id_Sensors in {ids} " \
                           f"and `TimeStamp` between {start_ts} and {end_ts}"
        self.cursor = self.cnx.cursor()
        self.cursor.execute(sql_select_query)
        records = self.cursor.fetchall()
        # create a index list and reindex dataframe
        idx = list()
        for row in records:
            timestamp = row[1]
            idx.append(timestamp)
            idx = list(set(idx))
            self.df = self.df.reindex(idx)
        # fill dataframe
        for row in records:
            timestamp = row[1]
            sensor_id = row[2]
            power = float(row[3])
            self.df.at[timestamp, sensor_id] = power

    def save_data(self, filename: str) -> None:
        """
        saves the data as csv
        :param filename: name of the csv file
        :return:
        """
        print(f"Save data to csv: {filename}")
        self.df.to_csv(f"data/database/{filename}.csv", sep=";")

    def close_connection(self) -> None:
        """
        close the database connection
        :return:
        """
        if self.cnx.is_connected():
            self.cnx.close()
            if self.cursor is not None:
                self.cursor.close()
            print("MySQL connection is closed")


def main():
    date_list = create_date_list()
    # set up database connection and dataframe
    db = SQLDatabase()
    db.establish_connection()
    db.get_sensor_ids()
    for d in date_list:
        start_time = time.time()
        start_dt, end_dt, filename = d
        # fetch historical data and save them
        db.get_history(start_dt, end_dt)
        db.save_data(filename)
        # calculate the execution time in seconds
        execution_time = int(time.time() - start_time)
        print(f"Execution time: {execution_time} s\n")
    db.close_connection()


if __name__ == "__main__":
    main()
