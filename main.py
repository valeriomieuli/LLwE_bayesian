import os
import sys
import numpy as np
from tensorflow.keras.utils import to_categorical

from warnings import filterwarnings

filterwarnings("ignore")

import utils
import utils_data
import utils_models
import utils_plots

####################################### Settings #######################################
datasets = ['scenes', 'birds', 'flowers', 'cars', 'aircrafts']
n_classes = {'scenes': 67, 'flowers': 102, 'birds': 200, 'cars': 196, 'aircrafts': 90}
trained_autoencoders, trained_experts = [], []

train_autoencs_acc_hist = {dataset: np.zeros(len(datasets) - 1) for dataset in datasets}
train_exps_acc_hist = {dataset: np.zeros(len(datasets) - 1) for dataset in datasets}
valid_autoencs_acc_hist = {dataset: np.zeros(len(datasets) - 1) for dataset in datasets}
valid_exps_acc_hist = {dataset: np.zeros(len(datasets) - 1) for dataset in datasets}
test_autoencs_acc_hist = {dataset: np.zeros(len(datasets) - 1) for dataset in datasets}
test_exps_acc_hist = {dataset: np.zeros(len(datasets) - 1) for dataset in datasets}

data_dir = sys.argv[1]
data_size = int(sys.argv[2])
valid_split = float(sys.argv[3])
seed = int(sys.argv[4])
test_split = float(sys.argv[5])

result_filename = "result.txt"
data_shape = (data_size, data_size, 3)
batch_size = 16

features_extractor_name = 'VGG16'
autoencoder_hidden_layer_sizes = [50, 100]  # 50, 100, 200]
autoencoder_weight_decays = [0.005, 0.0005]
autoencoder_learning_rates = [0.1, 0.01]  # [0.1, 0.01, 0.001]
autoencoder_epsilons = [1e-07, 1e-08]
autoencoder_objective_loss = 'binary_crossentropy'
autoencoder_epochs = 25

expert_name = 'InceptionResNetV2'

expert_weight_decays = {'scenes': [0.0005], 'flowers': [0.0005], 'birds': [0.005],
                        'cars': [0.0005], 'aircrafts': [0.005]}
expert_learning_rates = {'scenes': [0.01], 'flowers': [0.01], 'birds': [0.001],
                         'cars': [0.01], 'aircrafts': [0.01]}
expert_epsilons = {'scenes': [1e-07], 'flowers': [1e-07], 'birds': [1e-07],
                   'cars': [1e-07], 'aircrafts': [1e-07]}
expert_epochs = 25
########################################################################################

features_mean = np.load(os.path.join(os.getcwd(), str(data_size) + '_mean.npy'))
features_std = np.load(os.path.join(os.getcwd(), str(data_size) + '_std.npy'))
features_extractor = utils_models.build_features_extractor(features_extractor_name, data_shape)

for task_id, dataset in enumerate(datasets):
    if task_id == 0:
        open_mode = 'w'
    else:
        open_mode = 'a'
    utils.write_on_file(result_filename, open_mode,
                        "[%s] Starting new task..." % dataset.upper())

    '''X_train, y_train, X_valid, y_valid = utils_data.load_data(data_dir=data_dir, dataset=dataset, data_size=data_size,
                                                          phase='train_valid', valid_split=valid_split, seed=seed)'''
    X_train, y_train, X_valid, y_valid, _, _ = utils_data.load_data(data_dir=data_dir, dataset=dataset,
                                                                    data_size=data_size,
                                                                    valid_split=valid_split, test_split=test_split,
                                                                    seed=seed)
    ############################################ TRAINING ############################################
    utils.write_on_file(result_filename, 'a',
                        "[%s] Starting autoencoder's cross-validation..." % dataset.upper())
    autoencoder = utils.autoencoder_cross_validation(features_extractor, batch_size, X_train, X_valid, features_mean,
                                                     features_std,
                                                     autoencoder_hidden_layer_sizes, autoencoder_weight_decays,
                                                     autoencoder_learning_rates, autoencoder_epsilons,
                                                     autoencoder_epochs, autoencoder_objective_loss, dataset)
    trained_autoencoders.append(autoencoder)

    utils.write_on_file(result_filename, 'a',
                        "[%s] Starting expert's cross-validation..." % dataset.upper())
    expert = utils.expert_cross_validation(expert_name, batch_size, X_train, y_train, X_valid, y_valid,
                                           n_classes[dataset],
                                           expert_weight_decays[dataset], expert_learning_rates[dataset],
                                           expert_epsilons[dataset], expert_epochs,
                                           dataset)
    trained_experts.append(expert)

    ############################################ TESTING ############################################
    if task_id > 0:
        utils.write_on_file(result_filename, 'a', '\n' + '#' * 40 + "TESTING" + '#' * 40)
        for t in range(task_id + 1):
            '''X_train, y_train, X_valid, y_valid = utils_data.load_data(data_dir=data_dir, dataset=datasets[t],
                                                                  data_size=data_size, phase='train_valid',
                                                                  valid_split=valid_split, seed=seed)'''
            X_train, y_train, X_valid, y_valid, X_test, y_test = utils_data.load_data(data_dir=data_dir,
                                                                                      dataset=datasets[t],
                                                                                      data_size=data_size,
                                                                                      valid_split=valid_split,
                                                                                      test_split=test_split,
                                                                                      seed=seed)
            ################## Train data ##################
            train_autoencoder_acc, guest_indexes = utils.compute_autoencoder_accuracy(X=X_train, task_id=t,
                                                                                      features_extractor=features_extractor,
                                                                                      features_mean=features_mean,
                                                                                      features_std=features_std,
                                                                                      trained_autoencoders=trained_autoencoders)
            train_expert_acc = utils.compute_expert_accuracy(trained_experts[t], X_train, y_train, guest_indexes)
            train_autoencs_acc_hist[datasets[t]][task_id - 1] = train_autoencoder_acc
            train_exps_acc_hist[datasets[t]][task_id - 1] = train_expert_acc
            utils.write_on_file(result_filename, 'a',
                                "[%s] TRAIN_AUTOENCODER_ACC=%f - TRAIN_EXPERT_ACC=%f" % (datasets[t].upper(),
                                                                                         train_autoencoder_acc,
                                                                                         train_expert_acc))
            ################## Valid data ##################
            valid_autoencoder_acc, guest_indexes = utils.compute_autoencoder_accuracy(X=X_valid, task_id=t,
                                                                                      features_extractor=features_extractor,
                                                                                      features_mean=features_mean,
                                                                                      features_std=features_std,
                                                                                      trained_autoencoders=trained_autoencoders)
            valid_expert_acc = utils.compute_expert_accuracy(trained_experts[t], X_valid, y_valid, guest_indexes)
            valid_autoencs_acc_hist[datasets[t]][task_id - 1] = valid_autoencoder_acc
            valid_exps_acc_hist[datasets[t]][task_id - 1] = valid_expert_acc
            utils.write_on_file(result_filename, 'a',
                                "[%s] VALID_AUTOENCODER_ACC=%f - VALID_EXPERT_ACC=%f" % (datasets[t].upper(),
                                                                                         valid_autoencoder_acc,
                                                                                         valid_expert_acc))

            ################## Test data ##################
            '''X_test, y_test = utils_data.load_data(data_dir=data_dir, dataset=datasets[t],
                                          data_size=data_size, phase='test')'''
            test_autoencoder_acc, guest_indexes = utils.compute_autoencoder_accuracy(X=X_test, task_id=t,
                                                                                     features_extractor=features_extractor,
                                                                                     features_mean=features_mean,
                                                                                     features_std=features_std,
                                                                                     trained_autoencoders=trained_autoencoders)
            test_expert_acc = utils.compute_expert_accuracy(trained_experts[t], X_test, y_test, guest_indexes)
            test_autoencs_acc_hist[datasets[t]][task_id - 1] = test_autoencoder_acc
            test_exps_acc_hist[datasets[t]][task_id - 1] = test_expert_acc
            utils.write_on_file(result_filename, 'a',
                                "[%s] TEST_AUTOENCODER_ACC=%f - TEST_EXPERT_ACC=%f\n" % (datasets[t].upper(),
                                                                                         test_autoencoder_acc,
                                                                                         test_expert_acc))

        utils.write_on_file(result_filename, 'a',
                            '\n' + '#' * 40 + '#' * 7 + '#' * 40 + '\n')

'''print(train_autoencs_acc_hist)
print(train_exps_acc_hist)
print(valid_autoencs_acc_hist)
print(valid_exps_acc_hist)
print(test_autoencs_acc_hist)
print(test_exps_acc_hist)'''

utils.write_on_file(result_filename, 'a', "[INFO] Building barcharts...")
utils_plots.accuracy_barchart(train_autoencs_acc_hist, 'autoencoders', 'train')
utils_plots.accuracy_barchart(train_exps_acc_hist, 'experts', 'train')
utils_plots.accuracy_barchart(valid_autoencs_acc_hist, 'autoencoders', 'valid')
utils_plots.accuracy_barchart(valid_exps_acc_hist, 'experts', 'valid')
utils_plots.accuracy_barchart(test_autoencs_acc_hist, 'autoencoders', 'test')
utils_plots.accuracy_barchart(test_exps_acc_hist, 'experts', 'test')
utils.write_on_file(result_filename, 'a', "[INFO] Successfully completed :)")
