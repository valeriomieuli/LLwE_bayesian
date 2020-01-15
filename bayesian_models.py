import os
import sys
from warnings import filterwarnings

filterwarnings("ignore")

import utils
import utils_data

####################################### Settings #######################################
datasets = ['scenes', 'birds', 'flowers', 'cars', 'aircrafts']
n_classes = {'scenes': 67, 'flowers': 102, 'birds': 200, 'cars': 196, 'aircrafts': 90}

data_dir = sys.argv[1]
data_size = int(sys.argv[2])
valid_split = float(sys.argv[3])
test_split = float(sys.argv[4])
seed = int(sys.argv[5])
ds = sys.argv[6]
base_model_name = sys.argv[7]

result_filename = "result.txt"
data_shape = (data_size, data_size, 3)
batch_size = 32
data_augmentation = False

weight_decays = [-1, 0.005]
learning_rates = [0.01, 0.001]
epsilons = [1e-07, 1e-08]
dropout_rates = [0.25, 0.35, 0.5]
epochs = 70
########################################################################################

bayesian_model = None
for task_id, dataset in enumerate(datasets):
    if ds == dataset:
        utils.write_on_file(result_filename, 'w', "[%s] Loading data..." % dataset.upper())
        print("[%s] Loading data..." % dataset.upper())
        X_train, y_train, X_valid, y_valid, X_test, y_test = utils_data.load_data(data_dir=data_dir,
                                                                                  dataset=dataset,
                                                                                  data_size=data_size,
                                                                                  valid_split=valid_split,
                                                                                  test_split=test_split,
                                                                                  seed=seed)

        utils.write_on_file(result_filename, 'a', "[%s] Normalizing data..." % dataset.upper())
        print("[%s] Normalizing data..." % dataset.upper())
        X_train = utils.normalize_data(base_model_name=base_model_name, X=X_train)
        X_valid = utils.normalize_data(base_model_name=base_model_name, X=X_valid)
        X_test = utils.normalize_data(base_model_name=base_model_name, X=X_test)

        utils.write_on_file(result_filename, 'a',
                            "[%s] Starting bayesian model's cross-validation..." % dataset.upper())
        print("[%s] Starting bayesian model's cross-validation..." % dataset.upper())
        bayesian_model = utils.bayesian_cross_validation(base_model_name, batch_size, X_train, y_train,
                                                         X_valid, y_valid, n_classes[dataset], weight_decays,
                                                         learning_rates, epsilons, epochs, dataset,
                                                         data_augmentation, dropout_rates, seed)

        utils.write_on_file(result_filename, 'a', "[%s] Saving best bayesian..." % dataset.upper())
        print("[%s] Saving best bayesian..." % dataset.upper())
        bayesian_model.save(os.path.join('./', dataset + '_' + str(data_size) + '_' + base_model_name + '_bayesian.h5'))

        utils.write_on_file(result_filename, 'a', '')
        print()

utils.write_on_file(result_filename, 'a', "[INFO] Done")
print("[INFO] Done")
