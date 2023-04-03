import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction import DictVectorizer
import scipy
from scipy import sparse
from sklearn import datasets
from array import array
from sklearn.model_selection import train_test_split

import os

PATH = './data/general_data/'

def svm_read_problem(data_file_name, return_scipy=False):
    """
    svm_read_problem(data_file_name, return_scipy=False) -> [y, x], y: list, x: list of dictionary
    svm_read_problem(data_file_name, return_scipy=True)  -> [y, x], y: ndarray, x: csr_matrix

    Read LIBSVM-format data from data_file_name and return labels y
    and data instances x.
    """
    if scipy != None and return_scipy:
        prob_y = array('d')
        prob_x = array('d')
        row_ptr = array('l', [0])
        col_idx = array('l')
    else:
        prob_y = []
        prob_x = []
        row_ptr = [0]
        col_idx = []
    indx_start = 1
    for i, line in enumerate(open(data_file_name)):
        line = line.split(None, 1)
        # In case an instance with all zero features
        if len(line) == 1: line += ['']
        label, features = line
        prob_y.append(float(label))
        if scipy != None and return_scipy:
            nz = 0
            for e in features.split():
                ind, val = e.split(":")
                if ind == '0':
                    indx_start = 0
                val = float(val)
                if val != 0:
                    col_idx.append(int(ind) - indx_start)
                    prob_x.append(val)
                    nz += 1
            row_ptr.append(row_ptr[-1] + nz)
        else:
            xi = {}
            for e in features.split():
                ind, val = e.split(":")
                xi[int(ind)] = float(val)
            prob_x += [xi]
    if scipy != None and return_scipy:
        prob_y = np.frombuffer(prob_y, dtype='d')
        prob_x = np.frombuffer(prob_x, dtype='d')
        col_idx = np.frombuffer(col_idx, dtype='l')
        row_ptr = np.frombuffer(row_ptr, dtype='l')
        prob_x = sparse.csr_matrix((prob_x, col_idx, row_ptr))
    return (prob_y, prob_x)


def load_data(dataset):
    # 数据集1  heart
    if dataset == 1:
        y, x = svm_read_problem(os.path.join(PATH, 'heart.txt'), return_scipy=False)
        random_state = 42
        dictvectorizer = DictVectorizer(sparse=False)
        X = dictvectorizer.fit_transform(x)
        y = np.array(y)
        X = np.array(X)

    # 数据集2  ionosphere
    elif dataset == 2:
        y, x = svm_read_problem(os.path.join(PATH, 'ionosphere.txt'), return_scipy=False)
        random_state = 42
        dictvectorizer = DictVectorizer(sparse=False)
        X = dictvectorizer.fit_transform(x)
        y = np.array(y)
        X = np.array(X)

    # 数据集3  breast
    elif dataset == 3:
        cancer = datasets.load_breast_cancer()
        X = cancer.data
        y = cancer.target
        random_state = 42
        for i in range(len(y)):
            if (y[i] == 0):
                y[i] = -1

    # 数据集4  Australian
    elif dataset == 4:
        y, x = svm_read_problem(os.path.join(PATH, 'Australian.txt'), return_scipy=False)
        random_state = 42
        dictvectorizer = DictVectorizer(sparse=False)
        X = dictvectorizer.fit_transform(x)
        y = np.array(y)
        X = np.array(X)

    # 数据集5  diabetes
    elif dataset == 5:
        y, x = svm_read_problem(os.path.join(PATH, 'diabetes.txt'), return_scipy=False)
        random_state = 42
        dictvectorizer = DictVectorizer(sparse=False)
        X = dictvectorizer.fit_transform(x)
        y = np.array(y)
        X = np.array(X)

    # 数据集6  phishing
    elif dataset == 6:
        y, x = svm_read_problem(os.path.join(PATH, 'german.numer.txt'), return_scipy=False)
        random_state = 42
        dictvectorizer = DictVectorizer(sparse=False)
        X = dictvectorizer.fit_transform(x)
        y = np.array(y)
        X = np.array(X)

    # 数据集7  a2a
    elif dataset == 7:
        y, x = svm_read_problem(os.path.join(PATH, 'a2a.txt'), return_scipy=False)
        random_state = 42
        dictvectorizer = DictVectorizer(sparse=False)
        X = dictvectorizer.fit_transform(x)
        y = np.array(y)
        X = np.array(X)
        for i in range(len(y)):
            if (y[i] == 2):
                y[i] = -1

    # 数据集8  mushrooms
    elif dataset == 8:
        y, x = svm_read_problem(os.path.join(PATH, 'mushrooms.txt'), return_scipy=False)
        random_state = 42
        dictvectorizer = DictVectorizer(sparse=False)
        X = dictvectorizer.fit_transform(x)
        y = np.array(y)
        X = np.array(X)
        for i in range(len(y)):
            if (y[i] == 2):
                y[i] = -1

    # 数据集9  phishing
    elif dataset == 9:
        y, x = svm_read_problem(os.path.join(PATH, 'phishing.txt'), return_scipy=False)
        random_state = 42
        dictvectorizer = DictVectorizer(sparse=False)
        X = dictvectorizer.fit_transform(x)
        y = np.array(y)
        X = np.array(X)
        for i in range(len(y)):
            if (y[i] == 0):
                y[i] = -1

    scaler = StandardScaler()  # 数据预处理，使得经过处理的数据符合正态分布，即均值为0，标准差为1
    X = scaler.fit_transform(X)

    return X, y, random_state

def load_manual_dataset():
    ''''''
    mean1, mean2 = np.array([-1, 2]), np.array([1, -1])
    mean3, mean4 = np.array([4, -4]), np.array([-4, 4])
    covar = np.array([[1.0, 0.8], [0.8, 1.0]])
    X1 = np.random.multivariate_normal(mean1, covar, 50)
    X1 = np.vstack((X1, np.random.multivariate_normal(mean3, covar, 50)))
    y1 = np.ones(X1.shape[0])
    X2 = np.random.multivariate_normal(mean2, covar, 50)
    X2 = np.vstack((X2, np.random.multivariate_normal(mean4, covar, 50)))
    y2 = -1 * np.ones(X2.shape[0])
    X_train = np.vstack((X1[:80], X2[:80]))
    y_train = np.hstack((y1[:80], y2[:80]))
    X_test = np.vstack((X1[80:], X2[80:]))
    y_test = np.hstack((y1[80:], y2[80:]))
    print('训练/测试集：')
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    print('----------------------------------------------------')
    return X_train, y_train, X_test, y_test