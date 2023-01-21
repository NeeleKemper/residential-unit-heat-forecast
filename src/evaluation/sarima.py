from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.pipeline.DataPipelineTS import DataPipelineTS
from src.utils.utils import parse_season
from src.evaluation.ts_evaluation import time_series_experiment

cfg_summer = {"p": 0, "d": 0, "q": 0, "P": 2, "D": 0, "Q": 0, "t": "c"}
cfg_winter = {"p": 2, "d": 0, "q": 0, "P": 1, "D": 1, "Q": 1, "t": "c"}
cfg_all = {"p": 0, "d": 1, "q": 2, "P": 2, "D": 0, "Q": 0, "t": "c"}


def main():
    season = parse_season()
    # load dataset
    dp = DataPipelineTS(season=season)
    series_train, series_test = dp()
    cfg = eval(f"cfg_{season}")
    model = SARIMAX(series_train, order=(cfg["p"], cfg["d"], cfg["q"]),
                    seasonal_order=(cfg["P"], cfg["D"], cfg["Q"], 24), trend=cfg["t"])

    experiment_name = f"sarima_{season}"
    time_series_experiment(model=model, experiment_name=experiment_name, season=season)


if __name__ == "__main__":
    main()
