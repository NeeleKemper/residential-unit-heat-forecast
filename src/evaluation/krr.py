from sklearn.kernel_ridge import KernelRidge

from src.utils.utils import parse_season
from src.evaluation.offline_evaluation import offline_experiment

cfg_summer = {"kernel": "rbf", "alpha": 0.1, "gamma": 3.2}
cfg_winter = {"kernel": "rbf", "alpha": 0.1, "gamma": 10.0}
cfg_all = {"kernel": "rbf", "alpha": 0.1, "gamma": 10.0}


def main():
    season = parse_season()
    cfg = eval(f"cfg_{season}")

    model = KernelRidge(kernel=cfg["kernel"], alpha=cfg["alpha"], gamma=cfg["gamma"])
    offline_experiment(model=model, model_type="sklearn", model_name="krr", season=season)


if __name__ == "__main__":
    main()
