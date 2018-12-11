import os

from sklearn.naive_bayes import GaussianNB

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
import pandas as pd
import numpy as np
from nltk.corpus import stopwords

import matplotlib.pyplot as plt

'''
CODE
'''

'''
load_data
v1,v2,,,
ham,"Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...",,,
ham,Ok lar... Joking wif u oni...,,,
spam,Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's,,,
'''

def load_data(filename):
    # file = open(filename, "r")
    # lines = csv.reader(file)

    dataset = pd.read_csv(filename, encoding='latin-1')
    dataset['label'] = dataset['label'].map({'ham': 0, 'spam': 1})


    # dataset = list(lines)
    # for i in range(1, len(dataset)):
    #     if dataset[i][0] == 'spam':
    #         dataset[i][0] = 1
    #     else:
    #         dataset[i][0] = 0
    return dataset


def split_dataset(dataset, split_ratio=0.67):
    all_messages = dataset['message'].shape[0]
    train_indexes, test_indexes = list(), list()

    for i in range(dataset.shape[0]):
        if np.random.uniform(0, 1) < split_ratio:
            train_indexes += [i]
        else:
            test_indexes += [i]
    train = dataset.loc[train_indexes]
    test = dataset.loc[test_indexes]
    train.reset_index(inplace=True)
    test.reset_index(inplace=True)

    print(train.head())
    print(test.head())

    return [train, test]


def test_spam_guassiannb_predictor():
    print("test_spam_guassiannb_predictor()")
    script_dir = os.path.dirname(__file__)
    filename = os.path.join(script_dir, "spam.csv")
    dataset = load_data(filename)
    print('Loaded data file {0} with {1} rows'.format(filename, len(dataset)))

    train_set, test_set = split_dataset(dataset)
    print('Split {0} rows into\n train_set {1}\n test_set with {2}'.format(len(dataset), len(train_set), len(test_set)))

    find_spam_words(train_set)

    model = train_classifier_gaussiannb(train_set)

    test_y = [tx.pop() for tx in test_set]
    predicted_y = model.predict(test_set)

    accuracy = eval_accuracy(predicted_y, test_y)
    print("accuracy: {0:0.2f}%".format(accuracy))


def find_spam_words(train_set):
    message_list = train_set[train_set['label'] == 1]['message'].values

    word_list = []

    spam_words = ' '.join(list(message_list))

    # tokenize ngram=1
    words = word_tokenize(spam_words)
    words = [w for w in words if len(w) > 2]

    sw = set(stopwords.words('english'))

    words = [w for w in words if not w in sw]

    # stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    for i in range(len(message_list)):
        message = message_list[i].lower()


        # # remove stopwords
        # sw =
        # words = [word for word in words if word not in sw]

        #stemming
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
        word_list.append(words)

    spam_words = ' '.join(list(message_list))
    stopwords = set(STOPWORDS)
    spam_wc = WordCloud(stopwords=stopwords).generate(text=spam_words)

    plt.imshow(spam_wc)
    plt.axis('off')
    plt.show()
    return message_col

def train_classifier_gaussiannb(train_set):
    y = np.array([a.pop() for a in train_set])
    x = np.array(train_set)
    model = GaussianNB()
    model.fit(x, y)
    return model


def eval_accuracy(prediction, actual):
    correct = 0
    for c in range(len(prediction)):
        if prediction[c] == actual[c]:
            correct += 1
        else:
            print("wrong c:{0} t:{1} p:{2}".format(c, actual[c], prediction[c]))
    accuracy = (correct / len(prediction)) * 100
    return accuracy


'''
MAIN
'''


try:
    print("start")
    test_spam_guassiannb_predictor()
except TypeError as err:
    print(err)