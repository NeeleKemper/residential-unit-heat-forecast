from src.utils.utils import parse_season
from src.offline.LSTM import LSTM
from src.evaluation.offline_evaluation import offline_experiment

cfg_summer = {'optimizer': 'adam', 'units_lstm': 512, 'activation_lstm': 'sigmoid',
              'recurrent_activation_lstm': 'linear', 'use_bias_lstm': 0, 'dropout_lstm': 0.1,
              'recurrent_dropout_lstm': 0.3, 'kernel_initializer_lstm': 'uniform',
              'recurrent_initializer_lstm': 'lecun_uniform', 'unit_forget_bias_lstm': 0,
              'recurrent_regularizer_lstm': 'l1_l2', 'kernel_regularizer_lstm': 'l2', 'bias_regularizer_lstm': 'l1_l2',
              'activity_regularizer_lstm': 'l1_l2', 'kernel_constraint_lstm': 'non_neg',
              'bias_constraint_lstm': 'min_max_norm', 'recurrent_constraint_lstm': 'unit_norm', 'go_backwards_lstm': 1,
              'time_major_lstm': 0, 'units_dense_1': 32, 'activation_dense_1': 'relu',
              'kernel_regularizer_dense_1': 'l2', 'activity_regularizer_dense_1': 'l1_l2',
              'bias_regularizer_dense_1': 'l1_l2', 'kernel_initializer_dense_1': 'he_uniform',
              'kernel_constraint_dense_1': 'non_neg', 'bias_constraint_dense_1': 'unit_norm', 'dropout_1': 0.0,
              'kernel_regularizer_dense_2': 'l2', 'activity_regularizer_dense_2': 'l1_l2',
              'bias_regularizer_dense_2': 'l1_l2', 'kernel_initializer_dense_2': 'glorot_normal',
              'kernel_constraint_dense_2': 'max_norm', 'bias_constraint_dense_2': 'non_neg', 'batch_size': 16}


cfg_winter = {'optimizer': 'adam', 'units_lstm': 64, 'activation_lstm': 'linear', 'recurrent_activation_lstm': 'relu',
              'use_bias_lstm': 1, 'dropout_lstm': 0.2, 'recurrent_dropout_lstm': 0.1,
              'kernel_initializer_lstm': 'glorot_uniform', 'recurrent_initializer_lstm': 'glorot_uniform',
              'unit_forget_bias_lstm': 1, 'recurrent_regularizer_lstm': 'l1', 'kernel_regularizer_lstm': 'l1_l2',
              'bias_regularizer_lstm': 'l1', 'activity_regularizer_lstm': 'l1',
              'kernel_constraint_lstm': 'min_max_norm', 'bias_constraint_lstm': 'non_neg',
              'recurrent_constraint_lstm': 'max_norm', 'go_backwards_lstm': 0, 'time_major_lstm': 0,
              'units_dense_1': 32, 'activation_dense_1': 'relu', 'kernel_regularizer_dense_1': 'l1',
              'activity_regularizer_dense_1': 'l2', 'bias_regularizer_dense_1': 'l2',
              'kernel_initializer_dense_1': 'he_normal', 'kernel_constraint_dense_1': 'non_neg',
              'bias_constraint_dense_1': 'min_max_norm', 'dropout_1': 0.4, 'kernel_regularizer_dense_2': 'l1_l2',
              'activity_regularizer_dense_2': 'l2', 'bias_regularizer_dense_2': 'l1',
              'kernel_initializer_dense_2': 'glorot_normal', 'kernel_constraint_dense_2': 'max_norm',
              'bias_constraint_dense_2': 'non_neg', 'batch_size': 16}

cfg_all = {'optimizer': 'adam', 'units_lstm': 512, 'activation_lstm': 'sigmoid', 'recurrent_activation_lstm': 'linear',
           'use_bias_lstm': 0, 'dropout_lstm': 0.1, 'recurrent_dropout_lstm': 0.3, 'kernel_initializer_lstm': 'uniform',
           'recurrent_initializer_lstm': 'lecun_uniform', 'unit_forget_bias_lstm': 0,
           'recurrent_regularizer_lstm': 'l1_l2', 'kernel_regularizer_lstm': 'l2', 'bias_regularizer_lstm': 'l1_l2',
           'activity_regularizer_lstm': 'l1_l2', 'kernel_constraint_lstm': 'non_neg',
           'bias_constraint_lstm': 'min_max_norm', 'recurrent_constraint_lstm': 'unit_norm', 'go_backwards_lstm': 1,
           'time_major_lstm': 0, 'units_dense_1': 32, 'activation_dense_1': 'relu', 'kernel_regularizer_dense_1': 'l2',
           'activity_regularizer_dense_1': 'l1_l2', 'bias_regularizer_dense_1': 'l1_l2',
           'kernel_initializer_dense_1': 'he_uniform', 'kernel_constraint_dense_1': 'non_neg',
           'bias_constraint_dense_1': 'unit_norm', 'dropout_1': 0.0, 'kernel_regularizer_dense_2': 'l2',
           'activity_regularizer_dense_2': 'l1_l2', 'bias_regularizer_dense_2': 'l1_l2',
           'kernel_initializer_dense_2': 'glorot_normal', 'kernel_constraint_dense_2': 'max_norm',
           'bias_constraint_dense_2': 'non_neg', 'batch_size': 16} 


def main():
    season = parse_season()
    cfg = eval(f"cfg_{season}")

    model = LSTM(parameters=cfg)
    offline_experiment(model=model, model_type="keras", model_name="lstm", season=season)


if __name__ == "__main__":
    main()
