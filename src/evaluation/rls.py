from src.online.RLS import FilterRLS

from src.utils.utils import parse_season
from src.evaluation.online_evaluation import online_experiment

cfg_summer = {"mu": 0.99, "eps": 0.01}
cfg_winter = {"mu": 0.99, "eps": 0.01}
cfg_all = {"mu": 0.99, "eps": 0.01}

def main():
    season = parse_season()
    cfg = eval(f"cfg_{season}")

    model = FilterRLS(n=8, mu=cfg["mu"], eps=cfg["eps"])
    online_experiment(model=model, model_type="filter", model_name="rls", season=season)


if __name__ == "__main__":
    main()

