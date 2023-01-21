from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.utils.utils import parse_season
from src.evaluation.offline_evaluation import offline_experiment

cfg_summer = {"kernel": "rbf", "C": 46500, "gamma": "scale", "epsilon": 100.0}
cfg_winter = {"kernel": "rbf", "C": 105000, "gamma": "scale", "epsilon": 52.2}
cfg_all = {"kernel": "rbf", "C": 105000, "gamma": "scale", "epsilon": 100}


def main():
    season = parse_season()
    cfg = eval(f"cfg_{season}")

    model = Pipeline([
        ('scale', StandardScaler()),
        ('clf', SVR(kernel=cfg["kernel"], C=cfg["C"], gamma=cfg["gamma"], epsilon=cfg["epsilon"]))])
    offline_experiment(model=model, model_type="sklearn", model_name="svr", season=season)


if __name__ == "__main__":
    main()
