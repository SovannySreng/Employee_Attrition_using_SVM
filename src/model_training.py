from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def train_logistic_regression(x_train, y_train):
    lg = LogisticRegression()
    lg.fit(x_train, y_train)
    return lg

def train_svm(x_train, y_train, kernel='linear', degree=3):
    svm = SVC(kernel=kernel, degree=degree)
    svm.fit(x_train, y_train)
    return svm