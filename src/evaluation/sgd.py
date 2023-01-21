from sklearn.linear_model import SGDRegressor

from src.utils.utils import parse_season
from src.evaluation.online_evaluation import online_experiment

cfg_summer = {"loss": "huber", "penalty": "l1", "learning_rate": "invscaling"}
cfg_winter = {"loss": "huber", "penalty": "l1", "learning_rate": "invscaling"}
cfg_all = {"loss": "huber", "penalty": "l2", "learning_rate": "invscaling"}


def main():
    season = parse_season()
    cfg = eval(f"cfg_{season}")

    model = SGDRegressor(loss=cfg["loss"], penalty=cfg["penalty"], learning_rate=cfg["learning_rate"])
    online_experiment(model=model, model_type="sklearn", model_name="sgd", season=season)


if __name__ == "__main__":
    main()



