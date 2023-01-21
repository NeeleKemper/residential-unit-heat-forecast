from src.online.HBP import HedgeBackPropagation
from src.utils.utils import parse_season
from src.evaluation.online_evaluation import online_experiment

cfg_summer = {'n_layers': 2, 'hidden_units': 8, 'activation': 'linear', 'optimizer': 'adam', 'learning_rate': 0.001,
              'kernel_initializer': 'glorot_normal', 'kernel_regularizer': 'l1_l2', 'activity_regularizer': 'l1',
              'bias_regularizer': 'l1_l2', 'kernel_constraint': 'non_neg', 'bias_constraint': 'min_max_norm'}

cfg_winter = {'n_layers': 2, 'hidden_units': 128, 'activation': 'relu', 'optimizer': 'adam', 'learning_rate': 0.001,
              'kernel_initializer': 'random_uniform', 'kernel_regularizer': 'l1_l2', 'activity_regularizer': 'l1',
              'bias_regularizer': 'l1_l2', 'kernel_constraint': 'max_norm', 'bias_constraint': 'min_max_norm'}

cfg_all = {'n_layers': 2, 'hidden_units': 64, 'activation': 'relu', 'optimizer': 'adam', 'learning_rate': 0.001,
           'kernel_initializer': 'random_uniform', 'kernel_regularizer': 'l1_l2', 'activity_regularizer': 'l1',
           'bias_regularizer': 'l1', 'kernel_constraint': 'min_max_norm', 'bias_constraint': 'min_max_norm'}


def main():
    season = parse_season()
    cfg = eval(f"cfg_{season}")

    model = HedgeBackPropagation(
        n_layers=cfg["n_layers"],
        hidden_units=cfg["hidden_units"],
        activation=cfg["activation"],
        out_activation="relu",
        kernel_initializer=cfg["kernel_initializer"],
        kernel_regularizer=cfg["kernel_regularizer"],
        activity_regularizer=cfg["activity_regularizer"],
        bias_regularizer=cfg["bias_regularizer"],
        kernel_constraint=cfg["kernel_constraint"],
        bias_constraint=cfg["bias_constraint"],
    )
    model.compile(optimizer=cfg["optimizer"], learning_rate=cfg["learning_rate"], metrics=["mse"])
    online_experiment(model=model, model_type="keras", model_name="hbp", season=season)


if __name__ == "__main__":
    main()
