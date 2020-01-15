import os
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from PIL import Image


def load_data(data_dir, dataset, data_size, valid_split, test_split, seed):
    files = os.listdir(os.path.join(data_dir, dataset))
    X, y = np.zeros((len(files), data_size, data_size, 3)), np.zeros(len(files))
    for i, f in enumerate(files):
        img = adjust_image_shape(Image.open(os.path.join(data_dir, dataset, f)), data_size)
        X[i, :, :, :] = np.asarray(img)
        y[i] = int((f.split('_')[0]))
    if valid_split == 0 and test_split == 0:
        return X, y
    else:
        X, y = shuffle(X, y, random_state=seed)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=seed)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_split,
                                                              random_state=seed)
        return X_train, y_train, X_valid, y_valid, X_test, y_test


def adjust_image_shape(img, data_size):
    img = np.asarray(img)
    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)
    if img.shape[-1] == 4:
        new_img = img[:, :, :3]
    else:
        new_img = img
    new_img = Image.fromarray(new_img)
    width, height = new_img.size
    if height >= data_size and width >= data_size:
        resized_img = new_img.resize((data_size, data_size))
    else:
        resized_img = Image.new("RGB", (data_size, data_size))
        resized_img.paste(new_img, (int((data_size - width) / 2), int((data_size - height) / 2)))
    return resized_img
