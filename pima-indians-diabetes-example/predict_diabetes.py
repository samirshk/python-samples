import csv
import os
import random

import numpy as np
from sklearn.naive_bayes import GaussianNB

'''
CODE
'''


def load_data(filename):
    file = open(filename, "r")
    lines = csv.reader(file)
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


def split_dataset(dataset, split_ratio):
    train_size = int(len(dataset) * split_ratio)
    train_set = []
    copy = list(dataset)
    while len(train_set) < train_size:
        index = random.randrange(len(copy))
        train_set.append(copy.pop(index))
    return [train_set, copy]


'''
TEST
'''


def test_split_dataset():
    print("test_split_dataset()")
    dataset = [[1], [2], [3], [4], [5]]
    split_ratio = 0.67
    train, test = split_dataset(dataset, split_ratio)
    print('Split {0} rows into\n train {1}\n test with {2}'.format(len(dataset), len(train), len(test)))


def test_load_data():
    print("test_load_data()")
    script_dir = os.path.dirname(__file__)
    filename = os.path.join(script_dir, "pima-indians-diabetes.data.csv")
    dataset = load_data(filename)
    print('Loaded data file {0} with {1} rows'.format(filename, len(dataset)))


def test_diabetes_predictor():
    print("test_diabetes_predictor()")
    script_dir = os.path.dirname(__file__)
    filename = os.path.join(script_dir, "pima-indians-diabetes.data.csv")
    dataset = load_data(filename)
    print('Loaded data file {0} with {1} rows'.format(filename, len(dataset)))

    split_ratio = 0.67
    train, test = split_dataset(dataset, split_ratio)
    print('Split {0} rows into\n train {1}\n test with {2}'.format(len(dataset), len(train), len(test)))

    print(np)

    Y = np.array([x.pop() for x in dataset])
    print(len(Y))
    X = np.array(dataset)
    print(len(X))

    #Create a Gaussian Classifier
    model = GaussianNB()

    model.fit(X, Y)

    prediction = model.predict([
        [7, 100, 0, 0, 0, 30.0, 0.484, 32],
        [0, 118, 84, 47, 230, 45.8, 0.551, 31],
        [7, 107, 74, 0, 0, 29.6, 0.254, 31],
        [1, 103, 30, 38, 83, 43.3, 0.183, 33]
    ])

    print("prediction is {0}".format(prediction))

'''
MAIN
'''


try:
    print("start")
    test_load_data()
    test_split_dataset()
    test_diabetes_predictor()
except TypeError as err:
    print(err)