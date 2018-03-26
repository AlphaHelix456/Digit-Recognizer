import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical

def load_train():
    return pd.read_csv('train.csv')


def load_test():
    return pd.read_csv('test.csv')


def separate_train(train):
    Y_train = train['label']
    X_train = train.drop(labels=['label'], axis=1)
    return (X_train, Y_train)


def preprocess_input(X, Y):
    X = normalize(X)
    X = reshape(X)
    Y = encode_to_one_hot_vector(Y)


def _normalize(X):
    return X.astype('float32') / 255


def _reshape(X):
    return X.values.reshape(-1, 28, 28, 1)


def _encode_to_one_hot_vector(Y):
    return to_categorical(Y, num_classes=10)

