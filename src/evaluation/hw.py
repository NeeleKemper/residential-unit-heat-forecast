from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.pipeline.DataPipelineTS import DataPipelineTS
from src.utils.utils import parse_season
from src.evaluation.ts_evaluation import time_series_experiment


cfg_summer = {"trend": None, "seasonal": "additive"}
cfg_winter = {"trend": "multiplicative", "seasonal": "additive"}
cfg_all = {"trend": None, "seasonal": "additive"}


def main():
    season = parse_season()
    # load dataset
    dp = DataPipelineTS(season=season)
    series_train, series_test = dp()
    cfg = eval(f"cfg_{season}")
    model = ExponentialSmoothing(series_train, seasonal_periods=24, trend=cfg["trend"], seasonal=cfg["seasonal"])
    experiment_name = f"hw_{season}"
    time_series_experiment(model=model, experiment_name=experiment_name, season=season)

if __name__ == "__main__":
    main()
