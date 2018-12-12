import os

from sklearn.naive_bayes import GaussianNB

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from collections import Counter

from array import array
import matplotlib.pyplot as plt



'''
CODE
'''

'''
Global
'''
num_features = 10


'''
load_data
v1,v2,,,
ham,"Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...",,,
ham,Ok lar... Joking wif u oni...,,,
spam,Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's,,,
'''

def load_data(filename):
    dataset = pd.read_csv(filename, encoding='latin-1')
    dataset['label'] = dataset['label'].map({'ham': 0, 'spam': 1})
    dataset['message_feature_vector'] = [np.zeros(num_features)]*len(dataset)
    print(dataset.head())
    dataset.reindex()
    return dataset


def split_dataset(dataset, split_ratio=0.67):
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
    filename = os.path.join(script_dir, "spam-tmp.csv")
    dataset = load_data(filename)
    print('Loaded data file {0} with {1} rows'.format(filename, len(dataset)))

    train_set, test_set = split_dataset(dataset)
    print('Split {0} rows into\n train_set {1}\n test_set with {2}'.format(len(dataset), len(train_set), len(test_set)))

    spam_words_to_id_map, spam_id_to_words_map = find_top_spam_words(train_set)

    for index, row in train_set.iterrows():
        # fv = [int(0)]*len(spam_id_to_words_map)
        # fv = np.ndarray(shape=(1,len(spam_id_to_words_map)), dtype=float)
        fv = np.zeros(num_features)
        # fv = row['message_feature_vector']
        msg = normalize_text(row['message'])
        for w in msg:
            if w in spam_words_to_id_map:
                fv[spam_words_to_id_map[w]] = 1
        train_set.at[index, 'message_feature_vector'] = fv

    model = train_classifier_gaussiannb(train_set)

    for index, row in test_set.iterrows():
        fv = np.zeros(num_features)
        msg = normalize_text(row['message'])
        for w in msg:
            if w in spam_words_to_id_map:
                fv[spam_words_to_id_map[w]] = 1
        test_set.at[index, 'message_feature_vector'] = fv

    test_y = [tx.pop() for tx in test_set]
    predicted_y = model.predict(test_set)

    accuracy = eval_accuracy(predicted_y, test_y)
    print("accuracy: {0:0.2f}%".format(accuracy))


def find_top_spam_words(train_set):
    message_list = train_set[train_set['label'] == 1]['message'].values

    spam_words = ' '.join(list(message_list))

    words = normalize_text(spam_words)

    word_counter = Counter(words)

    words = word_counter.most_common(num_features)

    spam_words_to_id_map = {}

    i = 0
    for w in words:
        if w[0] in spam_words_to_id_map:
            continue
        else:
            spam_words_to_id_map[w[0]] = i
            i += 1

    spam_id_to_word_map = { }
    for key in spam_words_to_id_map:
        spam_id_to_word_map[spam_words_to_id_map[key]] = key

    return spam_words_to_id_map, spam_id_to_word_map


def normalize_text(spam_words):
    # tokenize ngram=1
    words = word_tokenize(spam_words)
    words = [w for w in words if len(w) > 2]
    sw = set(stopwords.words('english'))
    words = [w for w in words if not w in sw]
    # stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return words


def train_classifier_gaussiannb(train_set):
    yt = np.array(train_set['label'])
    # for y in range(yt):
    #     if len(yt[y]) > 1:
    #         print("error: " + yt[y])
    xt = np.ndarray(train_set['message_feature_vector'])
    print (xt.shape)
    for i in train_set['message_feature_vector']:
        print (len(i))

    for x in range(len(xt)):
        if len(xt[x]) != num_features:
            print("error: " + xt[x])

    train_set.to_csv("mmm.csv")
    model = GaussianNB()
    model.fit(xt, yt)
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