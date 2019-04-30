import datetime
import os


from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from collections import Counter
from nltk import ngrams
from nltk.stem import WordNetLemmatizer


'''
CODE
'''

'''
Global
'''
num_top_words = 500 #num_features
split_ratio = 0.75 #train/test split
gram = 1

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
    dataset['message_feature_vector'] = [np.zeros(num_top_words)] * len(dataset)
    dataset.reindex()
    return dataset


def split_dataset(dataset):
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

    return [train, test]

def write_af_file(dataset):
    dataset['qaid'] = dataset.apply(lambda x: 'spam.csv_'+str(x.name), axis=1)
    dataset['label'] = dataset['label'].replace('spam', '2')
    dataset.to_csv(path_or_buf='train_spam.csv', index=None, columns=['qaid', 'message', 'label', 'attrib1'])


def run_spam_predictor(classifier = 'nb rf'):
    script_dir = os.path.dirname(__file__)
    filename = os.path.join(script_dir, "spam.csv")
    dataset = load_data(filename)
    print('Loaded data file {0} with {1} rows'.format(filename, len(dataset)))

    write_af_file(dataset)

    train_set, test_set = split_dataset(dataset)
    print('Split {0} rows into\n train_set {1}\n test_set with {2}'.format(len(dataset), len(train_set), len(test_set)))

    spam_words_to_id_map, spam_id_to_words_map = find_top_spam_words(train_set)
    print('top spam words {0}'.format(spam_words_to_id_map.keys()))

    train_data_x = [np.zeros(num_top_words)] * len(train_set)
    train_data_y = np.zeros((len(train_set), 1))
    train_i = 0
    for index, row in train_set.iterrows():
        fv = np.zeros(num_top_words)
        msg = normalize_text(row['message'])
        for w in msg:
            if w in spam_words_to_id_map:
                fv[spam_words_to_id_map[w]] = 1
        train_set.at[index, 'message_feature_vector'] = fv
        train_data_x[train_i] = fv
        train_data_y[train_i] = row['label']
        train_i += 1

    test_data_x = [np.zeros(num_top_words)] * len(test_set)
    test_data_y = np.zeros((len(test_set), 1))
    test_i = 0
    for index, row in test_set.iterrows():
        fv = np.zeros(num_top_words)
        msg = normalize_text(row['message'])
        for w in msg:
            if w in spam_words_to_id_map:
                fv[spam_words_to_id_map[w]] = 1
        test_set.at[index, 'message_feature_vector'] = fv
        test_data_x[test_i] = fv
        test_data_y[test_i] = row['label']
        test_i += 1

    print("\n\n\n****Run Classifiers****\n")
    max_acc = 0.0
    winning_classifier = ''
    for c in classifier.split():
        model = train_classifier(train_data_x, train_data_y, c)
        start = datetime.datetime.now()
        predicted_y = model.predict(test_data_x)
        runtime = datetime.datetime.now()-start
        print('speed: {0:0.2f}/msg'.format(runtime.microseconds/len(test_data_y)))
        accuracy = eval_accuracy(predicted_y, test_data_y)
        print("accuracy: {0:0.2f}%".format(accuracy))
        if max_acc < accuracy:
            max_acc = accuracy
            winning_classifier = c
    print("\nClassifier {0} more accurate\n\n\n".format(winning_classifier))


def find_top_spam_words(train_set):
    global num_top_words

    message_list = train_set[train_set['label'] == 1]['message'].values

    spam_words = ' '.join(list(message_list))

    words = normalize_text(spam_words)

    word_counter = Counter(words)

    words = word_counter.most_common(num_top_words)

    if len(words) != num_top_words:
        print("force change num_top_words to " + str(len(words)))
        num_top_words = len(words)

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
    global gram
    words = word_tokenize(spam_words)
    words = [w.lower() for w in words if len(w) > 2]

    if gram > 1:
        sixgrams = ngrams(words, gram)
        words = [w for w in sixgrams]
        return words

    sw = set(stopwords.words('english'))
    words = [w for w in words if not w in sw]

    # # stemming
    # stemmer = PorterStemmer()
    # words = [stemmer.stem(word) for word in words]

    # lemmatize
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return words


def train_classifier(train_x, train_y, classifier='nb'):
    if classifier == 'nb':
        return train_classifier_nb(train_x, train_y)
    elif classifier == 'rf':
        return train_classifier_rf(train_x, train_y)
    else:
        raise Exception('no such classifier {0}'.format(classifier))

'''
Classifiers
'''
from sklearn.ensemble import RandomForestClassifier


def train_classifier_rf(train_x, train_y):
    print("\n***Random Forest Classifier***")
    model = RandomForestClassifier(n_estimators=100)
    model.fit(train_x, train_y)
    return model

from sklearn.naive_bayes import GaussianNB


def train_classifier_nb(train_x, train_y):
    print("***Naive Bayes Classifier***")
    model = GaussianNB()
    model.fit(train_x, train_y)
    return model


def eval_accuracy(prediction, actual):
    correct = 0
    for c in range(len(prediction)):
        if prediction[c] == actual[c]:
            correct += 1
        # else:
        #     print("wrong c:{0} t:{1} p:{2}".format(c, actual[c], prediction[c]))
    accuracy = (correct / len(prediction)) * 100
    return accuracy


'''
MAIN
'''


# for i in range(1,4):
#     gram = i
#     num_top_words = 500
#     print('ngram={0} num_top_words={1}'.format(i, num_top_words))
run_spam_predictor()
