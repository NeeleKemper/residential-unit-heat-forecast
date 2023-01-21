import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.ensemble import IsolationForest
from src.utils.utils import split_season
from src.pipeline.DataPipelineOffline import DataPipelineOffline


def get_data() -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """

    :return:
    """
    dp = DataPipelineOffline(season="all")
    X, y = dp()
    df = dp.df
    return X, y, df


def estimate_anomalies() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """

    :return:
    """
    X, y, df = get_data()
    ms = len(y)
    iso = IsolationForest(n_estimators=500, contamination='auto', random_state=42, max_samples=ms, max_features=8)
    anomalies = iso.fit_predict(X, y)
    df["anomalies"] = anomalies
    df_summer = split_season(season="summer", df=df)
    df_winter = split_season(season="winter", df=df)
    return df, df_summer, df_winter


def main():
    df, df_summer, df_winter = estimate_anomalies()
    anomalies, anomalies_summer, anomalies_winter = df["anomalies"], df_summer["anomalies"], df_winter["anomalies"]
    n_anomaly = len(np.where(anomalies < 0)[0])
    n_anomaly_summer = len(np.where(anomalies_summer < 0)[0])
    n_anomaly_winter = len(np.where(anomalies_winter < 0)[0])
    per = round((n_anomaly / len(anomalies)) * 100, 2)
    per_summer = round(((n_anomaly_summer * 100) / n_anomaly), 2)
    per_winter = round(((n_anomaly_winter * 100) / n_anomaly), 2)

    print(f"Number of anomalies: {n_anomaly}, {per}% of data points are anomalies.")
    print(f"Summer: {n_anomaly_summer}, {per_summer}% f of the anomalies are in the summer months.")
    print(f"Winter: {n_anomaly_winter}, {per_winter}% f of the anomalies are in the winter months.")


if __name__ == "__main__":
    main()
