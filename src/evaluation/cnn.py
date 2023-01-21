from src.utils.utils import parse_season
from src.offline.CNN import CNN
from src.evaluation.offline_evaluation import offline_experiment

cfg_summer = {'optimizer': 'adam', 'filter_cnn_1': 128, 'kernel_size_cnn_1': 9, 'activation_cnn_1': 'relu',
              'kernel_regularizer_cnn_1': 'l2', 'activity_regularizer_cnn_1': 'l1_l2',
              'bias_regularizer_cnn_1': 'l1_l2', 'kernel_initializer_cnn_1': 'uniform',
              'kernel_constraint_cnn_1': 'max_norm', 'bias_constraint_cnn_1': 'min_max_norm', 'dropout_1': 0.1,
              'filter_cnn_2': 64, 'kernel_size_cnn_2': 3, 'activation_cnn_2': 'relu', 'kernel_regularizer_cnn_2': 'l2',
              'activity_regularizer_cnn_2': 'l1', 'bias_regularizer_cnn_2': 'l1_l2',
              'kernel_initializer_cnn_2': 'glorot_normal', 'kernel_constraint_cnn_2': 'unit_norm',
              'bias_constraint_cnn_2': 'max_norm', 'dropout_2': 0.1, 'filter_cnn_3': 128, 'kernel_size_cnn_3': 3,
              'activation_cnn_3': 'relu', 'kernel_regularizer_cnn_3': 'l1', 'activity_regularizer_cnn_3': 'l1',
              'bias_regularizer_cnn_3': 'l2', 'kernel_initializer_cnn_3': 'uniform',
              'kernel_constraint_cnn_3': 'unit_norm', 'bias_constraint_cnn_3': 'max_norm', 'dropout_3': 0.0,
              'units_dense_1': 16, 'activation_dense_1': 'relu', 'kernel_regularizer_dense_1': 'l1',
              'activity_regularizer_dense_1': 'l1_l2', 'bias_regularizer_dense_1': 'l1',
              'kernel_initializer_dense_1': 'normal', 'kernel_constraint_dense_1': 'unit_norm',
              'bias_constraint_dense_1': 'min_max_norm', 'kernel_regularizer_dense_2': 'l1',
              'activity_regularizer_dense_2': 'l1_l2', 'bias_regularizer_dense_2': 'l1_l2',
              'kernel_initializer_dense_2': 'uniform', 'kernel_constraint_dense_2': 'max_norm',
              'bias_constraint_dense_2': 'non_neg', 'batch_size': 16}

cfg_winter = {'optimizer': 'nadam', 'filter_cnn_1': 64, 'kernel_size_cnn_1': 9, 'activation_cnn_1': 'relu',
              'kernel_regularizer_cnn_1': 'l1', 'activity_regularizer_cnn_1': 'l1_l2',
              'bias_regularizer_cnn_1': 'l1_l2', 'kernel_initializer_cnn_1': 'lecun_uniform',
              'kernel_constraint_cnn_1': 'min_max_norm', 'bias_constraint_cnn_1': 'non_neg', 'dropout_1': 0.0,
              'filter_cnn_2': 128, 'kernel_size_cnn_2': 5, 'activation_cnn_2': 'linear',
              'kernel_regularizer_cnn_2': 'l1_l2', 'activity_regularizer_cnn_2': 'l1',
              'bias_regularizer_cnn_2': 'l1_l2', 'kernel_initializer_cnn_2': 'he_uniform',
              'kernel_constraint_cnn_2': 'max_norm', 'bias_constraint_cnn_2': 'min_max_norm', 'dropout_2': 0.0,
              'filter_cnn_3': 128, 'kernel_size_cnn_3': 9, 'activation_cnn_3': 'relu', 'kernel_regularizer_cnn_3': 'l1',
              'activity_regularizer_cnn_3': 'l1', 'bias_regularizer_cnn_3': 'l1_l2',
              'kernel_initializer_cnn_3': 'lecun_uniform', 'kernel_constraint_cnn_3': 'unit_norm',
              'bias_constraint_cnn_3': 'min_max_norm', 'dropout_3': 0.0, 'units_dense_1': 128,
              'activation_dense_1': 'relu', 'kernel_regularizer_dense_1': 'l2', 'activity_regularizer_dense_1': 'l2',
              'bias_regularizer_dense_1': 'l1', 'kernel_initializer_dense_1': 'lecun_uniform',
              'kernel_constraint_dense_1': 'non_neg', 'bias_constraint_dense_1': 'max_norm',
              'kernel_regularizer_dense_2': 'l1', 'activity_regularizer_dense_2': 'l1_l2',
              'bias_regularizer_dense_2': 'l1_l2', 'kernel_initializer_dense_2': 'he_uniform',
              'kernel_constraint_dense_2': 'unit_norm', 'bias_constraint_dense_2': 'non_neg', 'batch_size': 16}

cfg_all = {'optimizer': 'adam', 'filter_cnn_1': 16, 'kernel_size_cnn_1': 9, 'activation_cnn_1': 'relu',
           'kernel_regularizer_cnn_1': 'l1', 'activity_regularizer_cnn_1': 'l2', 'bias_regularizer_cnn_1': 'l1_l2',
           'kernel_initializer_cnn_1': 'lecun_uniform', 'kernel_constraint_cnn_1': 'min_max_norm',
           'bias_constraint_cnn_1': 'non_neg', 'dropout_1': 0.0, 'filter_cnn_2': 64, 'kernel_size_cnn_2': 5,
           'activation_cnn_2': 'linear', 'kernel_regularizer_cnn_2': 'l1', 'activity_regularizer_cnn_2': 'l1',
           'bias_regularizer_cnn_2': 'l2', 'kernel_initializer_cnn_2': 'lecun_uniform',
           'kernel_constraint_cnn_2': 'non_neg',
           'bias_constraint_cnn_2': 'max_norm', 'dropout_2': 0.2, 'filter_cnn_3': 128, 'kernel_size_cnn_3': 9,
           'activation_cnn_3': 'relu', 'kernel_regularizer_cnn_3': 'l1', 'activity_regularizer_cnn_3': 'l1',
           'bias_regularizer_cnn_3': 'l2', 'kernel_initializer_cnn_3': 'lecun_uniform',
           'kernel_constraint_cnn_3': 'unit_norm',
           'bias_constraint_cnn_3': 'min_max_norm', 'dropout_3': 0.0, 'units_dense_1': 2048,
           'activation_dense_1': 'linear',
           'kernel_regularizer_dense_1': 'l1_l2', 'activity_regularizer_dense_1': 'l1',
           'bias_regularizer_dense_1': 'l1_l2',
           'kernel_initializer_dense_1': 'lecun_uniform', 'kernel_constraint_dense_1': 'unit_norm',
           'bias_constraint_dense_1': 'max_norm', 'kernel_regularizer_dense_2': 'l1',
           'activity_regularizer_dense_2': 'l2', 'bias_regularizer_dense_2': 'l1_l2',
           'kernel_initializer_dense_2': 'lecun_uniform', 'kernel_constraint_dense_2': 'unit_norm',
           'bias_constraint_dense_2': 'non_neg', 'batch_size': 16}


def main():
    season = parse_season()
    cfg = eval(f"cfg_{season}")

    model = CNN(parameters=cfg)
    offline_experiment(model=model, model_type="keras", model_name="cnn", season=season)


if __name__ == "__main__":
    main()
