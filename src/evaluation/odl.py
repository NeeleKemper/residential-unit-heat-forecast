from src.online.ODL import ODL
from src.utils.utils import parse_season
from src.evaluation.online_evaluation import online_experiment

cfg_summer = {'optimizer': 'adam', 'learning_rate': 0.001, 'units_1': 512, 'activation_1': 'relu',
              'kernel_regularizer_1': 'l2', 'activity_regularizer_1': 'l1', 'bias_regularizer_1': 'l1',
              'kernel_initializer_1': 'random_uniform', 'kernel_constraint_1': 'non_neg',
              'bias_constraint_1': 'min_max_norm', 'units_2': 512, 'activation_2': 'linear',
              'kernel_regularizer_2': 'l2', 'activity_regularizer_2': 'l2', 'bias_regularizer_2': 'l2',
              'kernel_initializer_2': 'normal', 'kernel_constraint_2': 'min_max_norm', 'bias_constraint_2': 'unit_norm',
              'kernel_regularizer_3': 'l2', 'activity_regularizer_3': 'l1_l2', 'bias_regularizer_3': 'l1_l2',
              'kernel_initializer_3': 'glorot_uniform', 'kernel_constraint_3': 'unit_norm',
              'bias_constraint_3': 'max_norm'}

cfg_winter = {'optimizer': 'adam', 'learning_rate': 0.001, 'units_1': 512, 'activation_1': 'relu',
              'kernel_regularizer_1': 'l2', 'activity_regularizer_1': 'l1', 'bias_regularizer_1': 'l1',
              'kernel_initializer_1': 'random_uniform', 'kernel_constraint_1': 'non_neg',
              'bias_constraint_1': 'min_max_norm', 'units_2': 512, 'activation_2': 'linear',
              'kernel_regularizer_2': 'l2', 'activity_regularizer_2': 'l2', 'bias_regularizer_2': 'l2',
              'kernel_initializer_2': 'normal', 'kernel_constraint_2': 'min_max_norm', 'bias_constraint_2': 'unit_norm',
              'kernel_regularizer_3': 'l2', 'activity_regularizer_3': 'l1_l2', 'bias_regularizer_3': 'l1_l2',
              'kernel_initializer_3': 'glorot_uniform', 'kernel_constraint_3': 'unit_norm',
              'bias_constraint_3': 'max_norm'}

cfg_all = {'optimizer': 'adam', 'learning_rate': 0.001, 'units_1': 512, 'activation_1': 'relu',
           'kernel_regularizer_1': 'l2', 'activity_regularizer_1': 'l1', 'bias_regularizer_1': 'l1',
           'kernel_initializer_1': 'random_uniform', 'kernel_constraint_1': 'non_neg',
           'bias_constraint_1': 'min_max_norm', 'units_2': 512, 'activation_2': 'linear', 'kernel_regularizer_2': 'l2',
           'activity_regularizer_2': 'l2', 'bias_regularizer_2': 'l2', 'kernel_initializer_2': 'normal',
           'kernel_constraint_2': 'min_max_norm', 'bias_constraint_2': 'unit_norm', 'kernel_regularizer_3': 'l2',
           'activity_regularizer_3': 'l1_l2', 'bias_regularizer_3': 'l1_l2', 'kernel_initializer_3': 'glorot_uniform',
           'kernel_constraint_3': 'unit_norm', 'bias_constraint_3': 'max_norm'}


def main():
    season = parse_season()
    cfg = eval(f"cfg_{season}")

    model = ODL(parameters=cfg)
    model.compile(optimizer=cfg["optimizer"], learning_rate=cfg["learning_rate"], metrics=["mse"])
    online_experiment(model=model, model_type="keras", model_name="odl", season=season)


if __name__ == "__main__":
    main()
