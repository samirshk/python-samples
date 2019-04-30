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
    dataset['label'] = dataset['label'].map({'ham': "0", 'spam': "2"})
    dataset.rename(columns={'message': 'answer'})
    dataset.reindex()
    return dataset


def write_af_train_file(dataset):
    dataset.to_csv(path_or_buf='train_spam.csv', index=None, sep='\t', columns=['qaid', 'message', 'label', 'attrib1'],
                   header=False)


def write_af_answers_file(dataset):
    dataset.to_csv(path_or_buf='spam_answers.csv', index=None, sep='\t', columns=['qaid', 'message'])


def run_spam_to_af():
    script_dir = os.path.dirname(__file__)
    filename = os.path.join(script_dir, "spam.csv")
    dataset = load_data(filename)
    print('Loaded data file {0} with {1} rows'.format(filename, len(dataset)))

    dataset['qaid'] = dataset.apply(lambda x: 'spam.csv_'+str(x.name), axis=1)
    write_af_answers_file(dataset)
    dataset_spam = dataset[dataset.label == "2"]
    write_af_train_file(dataset_spam)


# for i in range(1,4):
#     gram = i
#     num_top_words = 500
#     print('ngram={0} num_top_words={1}'.format(i, num_top_words))
run_spam_to_af()
