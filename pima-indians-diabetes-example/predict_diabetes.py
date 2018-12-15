import csv
import os
import random

import numpy as np
from sklearn.naive_bayes import GaussianNB

import pprint
import math
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


def split_dataset(dataset, split_ratio=0.67):
    train_size = int(len(dataset) * split_ratio)
    train_set = []
    test_set = list(dataset)
    while len(train_set) < train_size:
        index = random.randrange(len(test_set))
        train_set.append(test_set.pop(index))
    return [train_set, test_set]


def eval_accuracy(prediction, actual):
    correct = 0
    for c in range(len(prediction)):
        if prediction[c] == actual[c]:
            correct += 1
        # else:
        #     print("wrong c:{0} t:{1} p:{2}".format(c, actual[c], prediction[c]))

    accuracy = (correct / len(prediction)) * 100
    return accuracy


<<<<<<< HEAD
=======

>>>>>>> 3fdd0f2ab6f525fc991ea46266a7c3059f20ba03
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


def test_diabetes_guassiannb_predictor():
    print("test_diabetes_predictor()")
    script_dir = os.path.dirname(__file__)
    filename = os.path.join(script_dir, "pima-indians-diabetes.data.csv")
    dataset = load_data(filename)
    print('Loaded data file {0} with {1} rows'.format(filename, len(dataset)))

    train_set, test_set = split_dataset(dataset)
    print('Split {0} rows into\n train_set {1}\n test_set with {2}'.format(len(dataset), len(train_set), len(test_set)))

    model = train_classifier_nb_manual(train_set)

    test_y = [tx.pop() for tx in test_set]
    # predicted_y = model.predict(test_set)

    predicted_y = predict(model, test_set)
    print('Prediction: {0}'.format(predicted_y))

    accuracy = eval_accuracy(predicted_y, test_y)
    print("accuracy: {0:0.2f}%".format(accuracy))


# def train_classifier_gaussiannb(train_set):
#     y = np.array([a.pop() for a in train_set])
#     x = np.array(train_set)
#     model = GaussianNB()
#     model.fit(x, y)
#     return model

'''
Manual NB Classifier
'''


def train_classifier_nb_manual(train_set):
    pp = pprint.PrettyPrinter(indent=4)
    seperated = separate_by_class(train_set)
    pp.pprint(seperated)
    summeries = summerize_by_seperated(seperated)
    pp.pprint(summeries)
    return summeries


def predict(summaries, input_set):
    predictions = []
    for i in range(len(input_set)):
        predictions.append(predict_one(summaries, input_set[i]))
    return predictions


def predict_one(summaries, input_vector):
    probabilities = calculate_class_probability(summaries, input_vector)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label


def calculate_class_probability(summaries, input_vector):
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = 1 # start with 100%
        for i in range(len(class_summaries)): # for each element in input_vector/cs
            mean, stdev = class_summaries[i]
            x = input_vector[i]
            probabilities[class_value] *= calculate_probability(x, mean, stdev)
    return probabilities


def calculate_probability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def summerize_by_seperated(seperated):
    summeries = {}
    ditems = seperated.items()
    for c, inst in ditems:
        summeries[c] = summerize(inst)
    return summeries


def summerize(dataset):
    pp = pprint.PrettyPrinter(indent=4)
    zd = zip(*dataset)
    pp.pprint(zd)
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zd]
    pp.pprint(summaries)
<<<<<<< HEAD
    del summaries[-1]  # remove class column
=======
    del summaries[-1] # remove class column
>>>>>>> 3fdd0f2ab6f525fc991ea46266a7c3059f20ba03
    return summaries


def separate_by_class(dataset):
    separated = {} # y : [[x],..]
    for i in range(len(dataset)):
        vector = dataset[i]
        if vector[-1] not in separated:
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


def mean(numbers):
    return sum(numbers) / len(numbers)


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)


'''
MAIN
'''


try:
    print("start")
    test_load_data()
    test_split_dataset()
    test_diabetes_guassiannb_predictor()
except TypeError as err:
    print(err)