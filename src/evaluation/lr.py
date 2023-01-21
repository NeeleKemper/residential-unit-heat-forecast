from sklearn.linear_model import LinearRegression

from src.utils.utils import parse_season
from src.evaluation.offline_evaluation import offline_experiment


def main():
    season = parse_season()
    model = LinearRegression()
    offline_experiment(model=model, model_type="sklearn", model_name="lr", season=season)


if __name__ == "__main__":
    main()
