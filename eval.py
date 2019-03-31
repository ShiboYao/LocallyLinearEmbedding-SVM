'''
Implemented by Shibo Yao, Mar 31 2019
Testify the PCA and LLE with linear SVM on linearly non-separable data
'''
import sys
import numpy as np
from util import *
from spectral import LLE
from scipy.linalg import svd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV



n_samples = 2000
noise = 0.1
seed = np.random.randint(1000)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Specify k and d!")
        exit(0)

    k = int(sys.argv[1])
    d = int(sys.argv[2])
    test_rt = 0.3

    twin = get_twin(n_samples, noise)
    moons = get_synthetic('moons', n_samples, noise)
    #circles = get_synthetic('circles', n_samples, noise)
    #roll = get_synthetic('roll', n_samples, noise)
    sphere = np.loadtxt('data/ionosphere.txt', delimiter=',', dtype=float)
    digits = np.loadtxt('data/digits.txt', delimiter=',', dtype=float)
    #news = get_20news(voc_size = 5000)
    #attface = get_ATTface(num_fold=40, num_img=10)

    data_dic = {'twin':twin,
                'moons':moons,
                #'circles':circles,
                #'roll':roll,
                'sphere':sphere,
                'digits':digits
                #'attface':attface,
                #'news':news
                }

    params = {'C':[0.05, 0.1, 0.5, 1, 5, 10, 20]}

    for key in data_dic:
        print(key)
        data = data_dic[key]
        X = data[:,:-1]
        y = data[:,-1]
        d_comp = d
        if d > X.shape[1]:
            d_comp = X.shape[1]
    
        print("LLE")
        comp_LLE = LLE(X, k, d_comp, n_jobs=-1, epsilon=1e-3)
        X_train, X_test, y_train, y_test = train_test_split(comp_LLE, y, test_size=test_rt, random_state=seed)
        clf = GridSearchCV(LinearSVC(), params, cv=5, n_jobs=-1, iid=True)
        clf.fit(X_train, y_train)
        print(clf.best_params_)
        y_hat = clf.predict(X_test)
        acc = sum(y_hat==y_test)/len(y_hat)
        print("Accuracy %.4f\n" %acc)

        print("PCA")
        comp_PCA = svd(X.T, full_matrices=False)[2].T[:,:d_comp]
        X_train, X_test, y_train, y_test = train_test_split(comp_PCA, y, test_size=test_rt, random_state=seed)
        clf = GridSearchCV(LinearSVC(), params, cv=5)
        clf.fit(X_train, y_train)
        print(clf.best_params_)
        y_hat = clf.predict(X_test)
        acc = sum(y_hat==y_test)/len(y_hat)
        print("Accuracy %.4f\n" %acc)

        
