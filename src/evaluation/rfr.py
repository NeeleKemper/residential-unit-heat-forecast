from sklearn.ensemble import RandomForestRegressor

from src.utils.utils import parse_season
from src.evaluation.offline_evaluation import offline_experiment


cfg_summer = {"n_estimators": 5000, "max_features": "log2", "min_samples_split": 2, "min_samples_leaf": 2,
              "min_weight_fraction_leaf": 0.0, "bootstrap": 1}
cfg_winter = {"n_estimators": 5000, "max_features": "log2", "min_samples_split": 2, "min_samples_leaf": 2,
              "min_weight_fraction_leaf": 0.0, "bootstrap": 0}
cfg_all = {"n_estimators": 5000, "max_features": "log2", "min_samples_split": 6, "min_samples_leaf": 2,
           "min_weight_fraction_leaf": 0.0, "bootstrap": 0}


def main():
    season = parse_season()
    cfg = eval(f"cfg_{season}")

    model = RandomForestRegressor(n_estimators=cfg["n_estimators"], max_features=cfg["max_features"],
                                  min_samples_split=cfg["min_samples_split"], min_samples_leaf=cfg["min_samples_leaf"],
                                  min_weight_fraction_leaf=cfg["min_weight_fraction_leaf"], bootstrap=cfg["bootstrap"])
    offline_experiment(model=model, model_type="sklearn", model_name="rfr", season=season)


if __name__ == "__main__":
    main()