from src.online.ODLER import ODLER
from src.utils.utils import parse_season
from src.evaluation.online_evaluation import online_experiment

cfg_summer = {"batch_size": 4, "capacity": 20000, "factor": 0.5, "min_recollection": 48, "optimizer": "adam",
              "learning_rate": 0.001}

cfg_winter = {"batch_size": 4, "capacity": 10000, "factor": 0.5, "min_recollection": 48, "optimizer": "adam",
              "learning_rate": 0.001}

cfg_all = {"batch_size": 16, "capacity": 20000, "factor": 0.5, "min_recollection": 0, "optimizer": "adam",
           "learning_rate": 0.001}


def main():
    season = parse_season()
    cfg = eval(f"cfg_{season}")

    model = ODLER(season=season, cfg=cfg)
    model.compile(optimizer=cfg["optimizer"], learning_rate=cfg["learning_rate"], metrics=["mse"])
    online_experiment(model=model, model_type="keras", model_name="odl_er", season=season)


if __name__ == "__main__":
    main()
