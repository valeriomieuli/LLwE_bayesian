import os
import sys
import math
# import random
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from warnings import filterwarnings

filterwarnings("ignore")

import utils
import utils_data
import utils_plots

####################################### Settings #######################################
datasets = ['scenes', 'birds', 'flowers', 'cars', 'aircrafts', ]
n_classes = {'scenes': 67, 'flowers': 102, 'birds': 200, 'cars': 196, 'aircrafts': 90}

data_dir = sys.argv[1]
data_size = int(sys.argv[2])
ds = sys.argv[3]
bayesian_model_name = sys.argv[4]
result_filename = "result.txt"
mc_samples = 100
batch_size = 4096
bayesians_dir = '/home/vmieuli/bayesians/'
########################################################################################

bayesian_file = '%s_%d_%s_bayesian.h5' % (ds, data_size, bayesian_model_name)
utils.write_on_file(result_filename, 'w', "[%s] Loading bayesian model: %s ..." % (ds.upper(), bayesian_file),
                    new_line=False)
bayesian_model = load_model(os.path.join(bayesians_dir, bayesian_file))
utils.write_on_file(result_filename, 'a', " Done")

all_std_uncertainties = {dataset: [] for dataset in datasets}
all_aleatoric_uncertainties = {dataset: [] for dataset in datasets}
all_epistemic_uncertainties = {dataset: [] for dataset in datasets}

# n_samples = min([len(os.listdir(os.path.join(data_dir, dataset))) for dataset in datasets])
for dataset in datasets:
    files = os.listdir(os.path.join(data_dir, dataset))
    '''random.shuffle(files)
    files = files[:n_samples]'''

    utils.write_on_file(result_filename, 'a', "[%s] Computing uncertainties..." % dataset.upper(), new_line=False)
    for batch_id in range(math.ceil(len(files) / batch_size)):
        start_id = batch_id * batch_size
        end_id = min((batch_id + 1) * batch_size, len(files))

        X = np.zeros(((end_id - start_id), data_size, data_size, 3))
        for f_id, f in enumerate(files[start_id:end_id]):
            img = utils_data.adjust_image_shape(Image.open(os.path.join(data_dir, dataset, f)), data_size)
            img = np.asarray(img)
            X[f_id, :, :, :] = img

        X = utils.normalize_data(base_model_name=bayesian_model_name, X=X)

        uncs = utils.compute_predictions_uncertainty(bayesian_model=bayesian_model, X=X, mc_samples=mc_samples)
        for j in range(len(X)):
            all_std_uncertainties[dataset].append(uncs[0][j])
            all_aleatoric_uncertainties[dataset].append(uncs[1][j])
            all_epistemic_uncertainties[dataset].append(uncs[2][j])

    utils.write_on_file(result_filename, 'a', " Done")

utils.write_on_file(result_filename, 'a', '')
labels = [d for d in datasets]
colors = ['blue', 'green', 'orange', 'red', 'purple']

uncertainty_type = 'std'
utils.write_on_file(result_filename, 'a', "[INFO] Building %s uncertainty charts..." % uncertainty_type, new_line=False)
np.save('%s_uncertainty.npy' % uncertainty_type, all_std_uncertainties)
utils_plots.plot_uncertainty_per_sample([all_std_uncertainties[k] for k in all_std_uncertainties.keys()],
                                        uncertainty_type, labels, bayesian_model_name, ds, colors)
utils.write_on_file(result_filename, 'a', " Done")

uncertainty_type = 'aleatoric'
utils.write_on_file(result_filename, 'a', "[INFO] Building %s uncertainty charts..." % uncertainty_type, new_line=False)
np.save('%s_uncertainty.npy' % uncertainty_type, all_aleatoric_uncertainties)
utils_plots.plot_uncertainty_per_sample([all_aleatoric_uncertainties[k] for k in all_aleatoric_uncertainties.keys()],
                                        uncertainty_type, labels, bayesian_model_name, ds, colors)
utils.write_on_file(result_filename, 'a', " Done")

uncertainty_type = 'epistemic'
utils.write_on_file(result_filename, 'a', "[INFO] Building %s uncertainty charts..." % uncertainty_type, new_line=False)
np.save('%s_uncertainty.npy' % uncertainty_type, all_epistemic_uncertainties)
utils_plots.plot_uncertainty_per_sample([all_epistemic_uncertainties[k] for k in all_aleatoric_uncertainties.keys()],
                                        uncertainty_type, labels, bayesian_model_name, ds, colors)
utils.write_on_file(result_filename, 'a', " Done")
