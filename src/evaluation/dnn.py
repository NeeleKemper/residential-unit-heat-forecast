from src.utils.utils import parse_season
from src.offline.DNN import DNN
from src.evaluation.offline_evaluation import offline_experiment

cfg_summer = {'optimizer': 'adam', 'units_1': 4096, 'activation_1': 'relu', 'kernel_regularizer_1': 'l1_l2',
              'activity_regularizer_1': 'l1_l2', 'bias_regularizer_1': 'l1', 'kernel_initializer_1': 'random_uniform',
              'kernel_constraint_1': 'unit_norm', 'bias_constraint_1': 'non_neg', 'dropout_1': 0.4, 'units_2': 4096,
              'activation_2': 'linear', 'kernel_regularizer_2': 'l1', 'activity_regularizer_2': 'l1_l2',
              'bias_regularizer_2': 'l1', 'kernel_initializer_2': 'random_uniform', 'kernel_constraint_2': 'unit_norm',
              'bias_constraint_2': 'min_max_norm', 'dropout_2': 0.4, 'kernel_regularizer_3': 'l1_l2',
              'activity_regularizer_3': 'l1_l2', 'bias_regularizer_3': 'l1_l2', 'kernel_initializer_3': 'he_uniform',
              'kernel_constraint_3': 'non_neg', 'bias_constraint_3': 'min_max_norm', 'batch_size': 64}

cfg_winter = {'optimizer': 'adam', 'units_1': 2048, 'activation_1': 'relu', 'kernel_regularizer_1': 'l1_l2',
              'activity_regularizer_1': 'l2', 'bias_regularizer_1': 'l1', 'kernel_initializer_1': 'lecun_uniform',
              'kernel_constraint_1': 'min_max_norm', 'bias_constraint_1': 'min_max_norm', 'dropout_1': 0.0,
              'units_2': 4096, 'activation_2': 'linear', 'kernel_regularizer_2': 'l1_l2',
              'activity_regularizer_2': 'l1', 'bias_regularizer_2': 'l1', 'kernel_initializer_2': 'random_uniform',
              'kernel_constraint_2': 'non_neg', 'bias_constraint_2': 'min_max_norm', 'dropout_2': 0.1,
              'kernel_regularizer_3': 'l1_l2', 'activity_regularizer_3': 'l1_l2', 'bias_regularizer_3': 'l1',
              'kernel_initializer_3': 'lecun_uniform', 'kernel_constraint_3': 'non_neg',
              'bias_constraint_3': 'unit_norm', 'batch_size': 128}

cfg_all = {'optimizer': 'adam', 'units_1': 4096, 'activation_1': 'linear', 'kernel_regularizer_1': 'l1',
           'activity_regularizer_1': 'l1', 'bias_regularizer_1': 'l1_l2', 'kernel_initializer_1': 'random_uniform',
           'kernel_constraint_1': 'min_max_norm', 'bias_constraint_1': 'unit_norm', 'dropout_1': 0.5, 'units_2': 4096,
           'activation_2': 'relu', 'kernel_regularizer_2': 'l1_l2', 'activity_regularizer_2': 'l1',
           'bias_regularizer_2': 'l1_l2', 'kernel_initializer_2': 'lecun_uniform', 'kernel_constraint_2': 'non_neg',
           'bias_constraint_2': 'min_max_norm', 'dropout_2': 0.5, 'kernel_regularizer_3': 'l1_l2',
           'activity_regularizer_3': 'l1', 'bias_regularizer_3': 'l1_l2', 'kernel_initializer_3': 'glorot_uniform',
           'kernel_constraint_3': 'min_max_norm', 'bias_constraint_3': 'unit_norm', 'batch_size': 16}


def main():

    season = parse_season()
    cfg = eval(f"cfg_{season}")

    model = DNN(parameters=cfg)
    offline_experiment(model=model, model_type="keras", model_name="dnn", season=season)


if __name__ == "__main__":
    main()
