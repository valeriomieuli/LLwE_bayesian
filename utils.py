import os
import tensorflow.keras as keras
import numpy as np
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from math import exp
from scipy.spatial.distance import euclidean
from tensorflow.keras.utils import to_categorical

import utils_models

from warnings import filterwarnings

filterwarnings("ignore")

result_filename = "result.txt"


def write_on_file(filename, open_mode, str, new_line=True):
    f = open(filename, open_mode)
    if new_line:
        str = str + '\n'
    f.write(str)
    f.close()


def standardize_features(features, mean, std):
    input = keras.layers.Input(shape=(features.shape[1],))
    output = keras.layers.Activation(activation='sigmoid')(input)
    model = keras.Model(inputs=input, outputs=output)

    standardized_features = model.predict((features - mean) / std)
    return standardized_features


def autoencoder_cross_validation(features_extractor, batch_size, X_train, X_valid, features_mean, features_std,
                                 hidden_layer_sizes, weight_decays, learning_rates, epsilons, epochs, objective_loss,
                                 dataset):
    best_model = None
    best_loss = None
    best_params = {'hidden_layer_size': None, 'weight_decay': None,
                   'learning_rate': None, 'epsilon': None}

    train_features = standardize_features(features_extractor.predict(X_train),
                                          features_mean, features_std)
    valid_features = standardize_features(features_extractor.predict(X_valid),
                                          features_mean, features_std)

    for hidden_layer_size in hidden_layer_sizes:
        for weight_decay in weight_decays:
            for learning_rate in learning_rates:
                for epsilon in epsilons:
                    start_time = time.time()
                    autoencoder = utils_models.build_autoencoder(train_features.shape[-1],
                                                                 hidden_layer_size, weight_decay)

                    autoencoder.compile(optimizer=keras.optimizers.Adagrad(lr=learning_rate, epsilon=epsilon),
                                        loss=objective_loss)

                    autoencoder.fit(train_features, train_features, batch_size=batch_size, epochs=epochs, verbose=0)

                    valid_loss = autoencoder.evaluate(valid_features, valid_features, verbose=0)

                    if best_loss is None or ((best_loss is not None) and valid_loss < best_loss):
                        best_model = autoencoder
                        best_loss = valid_loss
                        best_params['hidden_layer_size'] = hidden_layer_size
                        best_params['weight_decay'] = weight_decay
                        best_params['learning_rate'] = learning_rate
                        best_params['epsilon'] = epsilon

                    write_on_file(result_filename, 'a',
                                  "[%s] valid_loss=%f | hidden_layer_size=%d - weight_decay=%f - learning_rate=%f - epsilon=%.8f | ETA=%f"
                                  % (dataset.upper(), valid_loss, hidden_layer_size,
                                     weight_decay, learning_rate, epsilon, time.time() - start_time))

    write_on_file(result_filename, 'a',
                  "\n[%s] Best model's params: %s\n" % (dataset.upper(), str(best_params)))

    return best_model


def compute_autoencoder_accuracy(X, task_id, features_extractor, trained_autoencoders, features_mean, features_std):
    features = standardize_features(features_extractor.predict(X),
                                    features_mean, features_std)

    reconstruction_errors = np.zeros((len(X), len(trained_autoencoders)))
    probabilities = np.zeros((len(X), len(trained_autoencoders)))
    guest_indexes = []
    for aut_id, autoencoder in enumerate(trained_autoencoders):
        decoded_features = autoencoder.predict(features)

        for j in range(len(X)):
            reconstruction_errors[j, aut_id] = euclidean(features[j, :], decoded_features[j, :])

    correct_predictions = 0
    for j in range(len(X)):
        denom = np.sum([exp(-err / 2) for err in reconstruction_errors[j]])
        for t, err in enumerate(reconstruction_errors[j]):
            probabilities[j, t] = exp(-err / 2) / denom
        if np.argmax(probabilities[j]) == task_id:
            correct_predictions += 1
            guest_indexes.append(True)
        else:
            guest_indexes.append(False)

    autoencoder_acc = float(correct_predictions / len(X))
    '''
    ritornare anche, per ogni input sample, l'indice indicando l'expert predetto

    per ogni expert far passare tutti i samples in input ad esso da un bayesiano e valutarne l'incertezza

    per i samples incerti spedire indietro agli autoencoder (escludendo quello di prima)

    '''
    return autoencoder_acc, guest_indexes


def expert_cross_validation(model_name, batch_size, X_train, y_train, X_valid, y_valid, n_classes,
                            weight_decays, learning_rates, epsilons, epochs, dataset, data_augmentation):
    if os.path.isfile(os.path.join('./', dataset + '_expert.h5')):
        best_model = os.path.join('./', dataset + '_expert.h5')
        write_on_file(result_filename, 'a',
                      "[%s] Best expert loaded from an existing .h5 file (cross validation has been skipped)." % dataset.upper())
        print(
            "[%s] Best expert loaded from an existing .h5 file (cross validation has been skipped)." % dataset.upper())
    else:
        best_model, best_acc = None, None
        best_params = {'weight_decay': None, 'learning_rate': None, 'epsilon': None}

        for weight_decay in weight_decays:
            for learning_rate in learning_rates:
                for epsilon in epsilons:
                    start_time = time.time()

                    expert = utils_models.build_expert(model_name, X_train[0].shape, n_classes, weight_decay)
                    expert.compile(optimizer=keras.optimizers.Adagrad(lr=learning_rate, epsilon=epsilon),
                                   loss='categorical_crossentropy', metrics=['accuracy'])

                    callbacks = [EarlyStopping(monitor='val_acc', mode='max', verbose=1,
                                               patience=4, restore_best_weights=True)]
                    if data_augmentation:
                        image_generator = ImageDataGenerator(rescale=1. / 255, rotation_range=15, width_shift_range=.15,
                                                             height_shift_range=.15, horizontal_flip=True,
                                                             vertical_flip=True)
                        expert.fit_generator(
                            image_generator.flow(X_train, to_categorical(y_train), batch_size=batch_size),
                            steps_per_epoch=2 * (X_train.shape[0] // batch_size), epochs=epochs, verbose=2,
                            validation_data=(X_valid, to_categorical(y_valid)), callbacks=callbacks)
                    else:
                        expert.fit(X_train, to_categorical(y_train), batch_size=batch_size, epochs=epochs,
                                   verbose=2, validation_data=(X_valid, to_categorical(y_valid)), callbacks=callbacks)

                    _, valid_acc = expert.evaluate(X_valid, to_categorical(y_valid), verbose=0)
                    if best_acc is None or ((best_acc is not None) and valid_acc > best_acc):
                        best_model, best_acc = expert, valid_acc
                        best_params['weight_decay'] = weight_decay
                        best_params['learning_rate'] = learning_rate
                        best_params['epsilon'] = epsilon
                    write_on_file(result_filename, 'a',
                                  "[%s] valid_acc=%f | weight_decay=%f - learning_rate=%f - epsilon=%.8f | ETA=%f"
                                  % (dataset.upper(), valid_acc, weight_decay, learning_rate, epsilon,
                                     time.time() - start_time))
                    print("[%s] valid_acc=%f | weight_decay=%f - learning_rate=%f - epsilon=%.8f | ETA=%f"
                          % (dataset.upper(), valid_acc, weight_decay, learning_rate, epsilon,
                             time.time() - start_time))

        write_on_file(result_filename, 'a', "\n[%s] Best params: %s\n" % (dataset.upper(), str(best_params)))
        print("\n[%s] Best params: %s\n" % (dataset.upper(), str(best_params)))

    return best_model


def bayesian_cross_validation(base_model_name, batch_size, X_train, y_train, X_valid, y_valid, n_classes,
                              weight_decays, learning_rates, epsilons, epochs, dataset, data_augmentation,
                              dropout_rates, seed):
    best_model = None
    best_acc = None
    best_loss = None
    best_params = {'weight_decay': None, 'learning_rate': None, 'epsilon': None, 'dropout_rate': None}

    for dropout_rate in dropout_rates:
        for weight_decay in weight_decays:
            for learning_rate in learning_rates:
                for epsilon in epsilons:
                    start_time = time.time()

                    bayesian_model = utils_models.build_bayesian_model(base_model_name, X_train[0].shape, n_classes,
                                                                       weight_decay, dropout_rate, seed)
                    bayesian_model.compile(optimizer=keras.optimizers.Adagrad(lr=learning_rate, epsilon=epsilon),
                                           loss='categorical_crossentropy', metrics=['accuracy'])

                    callbacks = [EarlyStopping(monitor='val_acc', mode='max', verbose=1,
                                               patience=100, restore_best_weights=True)]
                    if data_augmentation:
                        image_generator = ImageDataGenerator(rescale=1. / 255, rotation_range=15, width_shift_range=.15,
                                                             height_shift_range=.15, horizontal_flip=True,
                                                             vertical_flip=True)
                        bayesian_model.fit_generator(
                            image_generator.flow(X_train, to_categorical(y_train), batch_size=batch_size),
                            steps_per_epoch=2 * (X_train.shape[0] // batch_size), epochs=epochs, verbose=2,
                            validation_data=(X_valid, to_categorical(y_valid)))
                    else:
                        bayesian_model.fit(X_train, to_categorical(y_train), batch_size=batch_size, epochs=epochs,
                                           verbose=2, validation_data=(X_valid, to_categorical(y_valid)),
                                           callbacks=callbacks)

                    valid_loss, valid_acc = bayesian_model.evaluate(X_valid, to_categorical(y_valid), verbose=0)

                    if best_acc is None or ((best_acc is not None) and valid_acc > best_acc):
                        best_model = bayesian_model
                        best_acc = valid_acc
                        best_loss = valid_loss
                        best_params['weight_decay'] = weight_decay
                        best_params['learning_rate'] = learning_rate
                        best_params['epsilon'] = epsilon
                        best_params['dropout_rate'] = dropout_rate

                    write_on_file(result_filename, 'a',
                                  "[%s] valid_acc=%f - valid_loss=%f | dropout_rate=%.2f - weight_decay=%f - learning_rate=%f - epsilon=%.8f | ETA=%f"
                                  % (dataset.upper(), valid_acc, valid_loss, dropout_rate, weight_decay, learning_rate,
                                     epsilon,
                                     time.time() - start_time))
                    print(
                        "[%s] valid_acc=%f - valid_loss=%f | dropout_rate=%.2f - weight_decay=%f - learning_rate=%f - epsilon=%.8f | ETA=%f"
                        % (dataset.upper(), valid_acc, valid_loss, dropout_rate, weight_decay, learning_rate, epsilon,
                           time.time() - start_time))

    write_on_file(result_filename, 'a',
                  "\n[%s] Best_acc=%f - Best_loss=%f | Best params: %s\n" % (
                      dataset.upper(), best_acc, best_loss, str(best_params)))
    print("\n[%s] Best_acc=%f - Best_loss=%f | Best params: %s\n" % (
        dataset.upper(), best_acc, best_loss, str(best_params)))

    return best_model


def compute_expert_accuracy(expert, X, y, guest_indexes):
    correct_predictions = 0
    y_filter = y[guest_indexes]
    predicted_probabilities = expert.predict(X[guest_indexes], verbose=0)
    for i in range(len(predicted_probabilities)):
        if np.argmax(predicted_probabilities[i]) == y_filter[i]:
            correct_predictions += 1
    # print('xx', len(X), len(X[guest_indexes]), correct_predictions)
    # loss, acc = expert.evaluate(X[guest_indexes], to_categorical(y[guest_indexes]), verbose=0)
    # print('xx', len(X), len(X[guest_indexes]), len(y[guest_indexes]), correct_predictions / len(X))
    return correct_predictions / len(X)


'''def compute_predictions_uncertainty(bayesian_model, X, n_classes, mc_samples, type):
    mc_predictions = np.zeros((mc_samples, X.shape[0], n_classes))

    if type == 'std':
        for t in range(mc_samples):
            mc_predictions[t, :, :] = bayesian_model.predict(X)

        uncertainties = np.zeros(len(X))
        for i in range(len(X)):
            mc_predicted_class = np.argmax(np.mean(mc_predictions[:, i, :], axis=0))
            uncertainties[i] = np.std(mc_predictions[:, i, mc_predicted_class], axis=0)
        return uncertainties

    elif type == 'aleatoric_epistemic':
        p_hat = []
        for t in range(mc_samples):
            p_hat.append(bayesian_model.predict(X))
        p_hat = np.array(p_hat)

        aleatoric = np.mean(p_hat * (1 - p_hat), axis=0)
        epistemic = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat, axis=0) ** 2
        predictions = np.mean(p_hat, axis=0)

        aleatoric_unc, epistemic_unc = np.zeros(len(X)), np.zeros(len(X))
        for i, x in enumerate(X):
            predicted_class = np.argmax(predictions[i])
            aleatoric_unc[i] = aleatoric[i, predicted_class]
            epistemic_unc[i] = epistemic[i, predicted_class]
        return [aleatoric_unc, epistemic_unc]

    else:
        raise ValueError("Type of uncertainty not available!")'''


def compute_predictions_uncertainty(bayesian_model, X, mc_samples):
    p_hat = []
    for t in range(mc_samples):
        p_hat.append(bayesian_model.predict(X))
    p_hat = np.array(p_hat)

    aleatoric = np.mean(p_hat * (1 - p_hat), axis=0)
    epistemic = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat, axis=0) ** 2
    predictions = np.mean(p_hat, axis=0)

    std_unc, aleatoric_unc, epistemic_unc = np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X))
    for i, x in enumerate(X):
        predicted_class = np.argmax(predictions[i])
        std_unc[i] = np.std(p_hat[:, i, predicted_class], axis=0)
        aleatoric_unc[i] = aleatoric[i, predicted_class]
        epistemic_unc[i] = epistemic[i, predicted_class]

    return std_unc, aleatoric_unc, epistemic_unc


def normalize_data(base_model_name, X):
    preprocessing_function = utils_models.get_preprocessing_function(model_name=base_model_name)
    return preprocessing_function(X)
