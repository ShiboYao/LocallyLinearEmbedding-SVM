'''
Shibo Yao, April 4 2019
Multiclass Support Vector Machine, Linear Kernel
Parallel training for pairwise hyperplanes
'''
import numpy as np
#from util import *
import multiprocessing as mp
from sklearn.datasets import load_breast_cancer, load_iris, load_digits, load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC



class SVM(object):
    def __init__(self, lr=0.001, C=1, num_batch=1000, batchSize=2014, tol=0.001, verbose=False):
        self.lr = lr
        self.C = C
        self.num_batch = num_batch
        self.batchSize = batchSize
        self.tol = tol
        self.verbose = verbose
        self.n_class = None
        self.w = None
        self.label2id = {}
        self.id2label = {}
        self.X = None
        self.y = None


    def SVMbase(self, index):
        """2-class problem"""
        y = self.y[index]
        y = y.reshape(-1,1)
        label_set = set(y.flat)
        label_list = [self.label2id[s] for s in label_set]
        label_list = sorted(label_list)
        print(label_list)
        y[y==label_list[0]] = -1
        y[y==label_list[1]] = 1

        X = self.X[index]
        n = X.shape[0]
        d = X.shape[1]+1
        one = np.ones([n,1])
        X = np.hstack((one,X)) #1,x1,x2,...xd

        if self.batchSize > n:
            self.batchSize = n

        w = np.random.rand(1,d)
        step = 0
        tol = 1+self.tol
        mem = 5
        err_pre = [np.array([np.inf])]*mem
    
        while step < self.num_batch:
            ind = np.random.choice(n,self.batchSize,replace=False)
            cost_ind = [i for i in ind if y[i]*np.inner(w,X[i])<1]
            w = (1-self.lr)*w + (y[cost_ind]*X[cost_ind]).sum(axis=0)*self.lr*self.C
            y_hat = np.dot(w,X.T)
            y_hat[y_hat>=0] = 1
            y_hat[y_hat<0] = -1
            y_hat = y_hat.reshape(-1,1)
            err = sum(y_hat!=y) / n

            if step%100==0:
                if self.verbose:
                    print("False rate: %.4f"%err)
                j = 0
                temp = err*tol
                while j < mem and temp>=err_pre[j]:
                    j = j+1
                if j == mem:
                    print("early stop")
                    return w
                else :
                    k = 0
                    while k < mem-1:
                        err_pre[k] = err_pre[k+1].copy()
                        k = k+1
                    err_pre[mem-1] = err.copy()
                #print(err_pre)       
            step = step+1

        return w


    def multiClassSVM(self):
        for i in range(self.n_class-1):
            ind_i = self.y==self.id2label[i]
            for j in range(i+1, self.n_class):
                ind_j = self.y==self.id2label[j]
                index = ind_i | ind_j
                self.w[i,j] = self.SVMbase(index)

        return


    def SVMbase_mp(self, index, return_dic):
        """2-class problem"""
        y = self.y[index]
        y = y.reshape(-1,1)
        label_set = set(y.flat)
        label_list = [self.label2id[s] for s in label_set]
        label_list = sorted(label_list)
        #print(label_list)
        y[y==label_list[0]] = -1
        y[y==label_list[1]] = 1

        X = self.X[index]
        n = X.shape[0]
        d = X.shape[1]+1
        one = np.ones([n,1])
        X = np.hstack((one,X)) #1,x1,x2,...xd

        if self.batchSize > n:
            self.batchSize = n

        w = np.random.rand(1,d)
        step = 0
        tol = 1+self.tol
        mem = 5
        err_pre = [np.array([np.inf])]*mem
    
        while step < self.num_batch:
            ind = np.random.choice(n,self.batchSize,replace=False)
            cost_ind = [i for i in ind if y[i]*np.inner(w,X[i])<1]
            w = (1-self.lr)*w + (y[cost_ind]*X[cost_ind]).sum(axis=0)*self.lr*self.C
            y_hat = np.dot(w,X.T)
            y_hat[y_hat>=0] = 1
            y_hat[y_hat<0] = -1
            y_hat = y_hat.reshape(-1,1)
            err = sum(y_hat!=y) / len(y)

            if step%100==0:
                if self.verbose:
                    print("False rate: %.4f"%err)
                j = 0
                temp = err*tol
                while j < mem and temp>=err_pre[j]:
                    j = j+1
                if j == mem:
                    #print("early stop")
                    return_dic[tuple(label_list)] = w
                else :
                    k = 0
                    while k < mem-1:
                        err_pre[k] = err_pre[k+1].copy()
                        k = k+1
                    err_pre[mem-1] = err.copy()
                #print(err_pre)       
            step = step+1

        #print(w)
        #self.w[label_list[0],label_list[1]] = w
        return_dic[tuple(label_list)] = w
        


    def multiCoreSVM(self):
        """parallel computing pairwise hyperplane"""
        index_list = []
        processes = []
        return_dic = mp.Manager().dict()
        n_jobs = 0

        for i in range(self.n_class-1):
            ind_i = self.y==self.id2label[i]
            for j in range(i+1,self.n_class):
                ind_j = self.y==self.id2label[j]
                index = ind_i | ind_j
                index_list.append(index)
                n_jobs += 1

        for i in range(n_jobs):
            proc = mp.Process(target=self.SVMbase_mp, args=(index_list[i], return_dic))
            processes.append(proc)
            proc.start()
        for process in processes:
            process.join()
        
        for i in range(self.n_class-1):
            for j in range(i+1,self.n_class):
                key = tuple([i,j])
                self.w[i,j] = return_dic[key]
        
        return


    def fit(self, X, y, parallel=False):
        self.X = X
        self.y = y
        d = X.shape[1]+1
        label_set = sorted(list(set(y.flat)))
        self.n_class = len(label_set)
        self.w = np.empty([self.n_class, self.n_class, d])
        i = 0
        for l in label_set:
            self.label2id[l] = i
            self.id2label[i] = l
            i += 1

        if parallel:
            self.multiCoreSVM()
        else :
            self.multiClassSVM()

        return 


    def predict(self, X_test):
        X_test = np.hstack((np.ones([X_test.shape[0],1]), X_test))
        y_hat = np.empty([X_test.shape[0], 1])
        
        for i in range(X_test.shape[0]):
            vote = [0.] * self.n_class
            for k in range(self.n_class-1):
                for j in range(k+1, self.n_class):
                    hat = np.inner(self.w[k,j], X_test[i])
                    if hat > 0: #multiclass inference, double check
                        vote[j] += 1
                    else :
                        vote[k] += 1
            #print(vote)
            y_hat[i] = self.id2label[np.argmax(vote)]

        return y_hat.reshape(-1)




seed=123

if __name__ == "__main__":
    #data = get_synthetic('moons', n_samples=1000, noise=0.1)
    #data = load_breast_cancer()
    #data = load_iris()
    data = load_digits()
    #data = load_wine()
    X = data.data
    y = data.target
    
    for i in range(X.shape[1]):#normalization, important to SVM
        X[:,i] = (X[:,i]-min(X[:,i]))/(max(X[:,i])+0.0001)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)
    '''
    clf = LinearSVC() #sklearn version
    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_test)
    '''
    clf = SVM(C=1, lr=0.001) #my version
    clf.fit(X_train, y_train, parallel=True)
    y_hat = clf.predict(X_test)
    
    print(y_hat)    
    acc = (y_hat==y_test).sum() / len(y_test)

    print("accuracy %.4f" %acc)


