from sklearn import metrics
from path import Path
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import pandas as pd

DATASET_PATH = Path()
SENTENCE_LIST = Path(r'NB_sentence_list.txt')
LABEL_LIST = Path(r'label_list.txt')


def evaluate_result(y_test, y_predicted, y_train):
    y_test = np.asarray(y_test)
    y_predicted = np.asarray(y_predicted)
    y_train = np.asarray(y_train)

    print('\n\nRESULTS:')

    print(metrics.classification_report(y_test, y_predicted, target_names=['negative', 'neutral', 'positive']))
    print('\nSuccess rate: ' + str(np.mean(y_predicted == y_test)))
    print('Dataset size = ' + str(len(y_train) + len(y_test)))

    print('\nTest set size: ' + str(len(y_test)))
    print('Train set size: ' + str(len(y_train)))

    print('\nNegative reviews count in TRAIN datatset:' + str(len(np.where(y_train == 0)[0])))
    print('Neutral reviews count in TRAIN datatset:' + str(len(np.where(y_train == 1)[0])))
    print('Positive reviews count in TRAIN datatset:' + str(len(np.where(y_train == 2)[0])))

    print('\nNegative reviews count in TEST datatset:' + str(len(np.where(y_test == 0)[0])))
    print('Neutral reviews count in TEST datatset:' + str(len(np.where(y_test == 1)[0])))
    print('Positive reviews count in TEST datatset:' + str(len(np.where(y_test == 2)[0])))

    print('\n Confusion matrix [negative, neutral, positive]')
    # print(metrics.confusion_matrix(y_test, y_predicted, labels=[0, 1, 2]))
    print(metrics.confusion_matrix(y_test, y_predicted))


def run_Multinomial_Naive_Bayes_ver_1(X_train, y_train, X_test):
    print('Training model')
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB(alpha=5).fit(X_train, y_train)

    print('Predicting on test data')
    predicted = clf.predict(X_test)
    predicted = np.asarray(predicted, dtype=np.uint64)

    return predicted


def run_Complement_Naive_Bayes(X_train, y_train, X_test, is_norm=False):
    print('Training model')
    from sklearn.naive_bayes import ComplementNB
    clf = ComplementNB(norm=is_norm).fit(X_train, y_train)

    print('Predicting on test data')
    predicted = clf.predict(X_test)
    predicted = np.asarray(predicted, dtype=np.uint64)

    return predicted


def run_my_MNNB(X_train, y_train, X_test):
    print('Training model')
    from my_MNNB import myMultinomialNB
    clf = myMultinomialNB()
    clf.fit(X_train, y_train)

    print('Predicting on test data')
    predicted = clf.predict(X_test)
    predicted = np.asarray(predicted, dtype=np.uint64)

    return predicted


def run_my_OVANB(X_train, y_train, X_test):
    print('Training model')
    from my_OVA_NB import myOVANB
    clf = myOVANB(alpha=0.2)
    clf.fit(X_train, y_train)

    print('Predicting on test data')
    predicted = clf.predict(X_test)
    predicted = np.asarray(predicted, dtype=np.uint64)

    return predicted


def run_my_WCNB(X_train, y_train, X_test):
    print('Training model')
    from my_WCNB import myWCNB
    clf = myWCNB(alpha=1e-1, fit_prior=False)
    clf.fit(X_train, y_train)

    print('Predicting on test data')
    predicted = clf.predict(X_test)
    predicted = np.asarray(predicted, dtype=np.uint64)

    return predicted


def run_SVM(X_train, y_train, X_test):
    print('Training model')
    from sklearn.linear_model import SGDClassifier

    clf = SGDClassifier()

    clf.fit(X_train, y_train)

    print('Predicting on test data')
    predicted = clf.predict(X_test)
    predicted = np.asarray(predicted, dtype=np.uint64)

    return predicted


if __name__ == '__main__':
    DATASET_PATH = Path('data/preprocessed_data/dataset_10k')

    print('Loading data')

    X = pd.read_json(DATASET_PATH.joinpath(SENTENCE_LIST))[0].tolist()
    y = pd.read_json(DATASET_PATH.joinpath(LABEL_LIST))[0].tolist()

    print('Splitting data')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=22)

    rus = RandomUnderSampler()
    X_train, y_train = rus.fit_resample(np.asarray(X_train).reshape(-1, 1), y_train)
    X_train = X_train.squeeze()
    y_train = y_train.squeeze()

    print('Transforming input data')
    # Transform train data
    from sklearn.feature_extraction.text import CountVectorizer

    count_vect = CountVectorizer(min_df=10, ngram_range=(1, 2))

    X_train_counts = count_vect.fit_transform(X_train)
    X_test_counts = count_vect.transform(X_test)

    # from sklearn.feature_extraction.text import TfidfTransformer
    # tfidf_transformer = TfidfTransformer(use_idf=True)
    #
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    # X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    # Transform test data
    X_train = X_train_counts
    X_test = X_test_counts

    # y_predicted = run_Multinomial_Naive_Bayes_ver_1(X_train, y_train, X_test)
    # y_predicted = run_Complement_Naive_Bayes(X_train, y_train, X_test, is_norm=True)
    y_predicted = run_my_OVANB(X_train, y_train, X_test)
    # y_predicted = run_my_WCNB(X_train, y_train, X_test)
    # y_predicted = run_SVM(X_train, y_train, X_test)

    evaluate_result(y_test, y_predicted, y_train)
