'''
Implemented by Shibo Yao, Mar 30 2019
Ref: Locally Linear Embedding, 2000 Science
'''
import sys
import numpy as np
from util import *
from scipy.linalg import eigh
import multiprocessing as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time



def dis_base(pid, index_list, return_dic, x):
    p = len(index_list)
    n = x.shape[0]
    small_dis = np.zeros([p, n])
    for i in range(p):
        ind = index_list[i]
        vec = x[ind]
        small_dis[i, (ind+1):n] = [sum((x[j]-vec)**2) for j in range(ind+1,n)] #in here no need to take square root as only order matters

    return_dic[pid] = small_dis


def multicore_dis(func, x, n_jobs=-1):
    """parallel distance matrix"""
    index_list = np.array_split(range(x.shape[0]), n_jobs)
    processes = []
    return_dic = mp.Manager().dict()

    for i in range(n_jobs):
        proc = mp.Process(target=func, args=(i,index_list[i],return_dic,x))
        processes.append(proc)
        proc.start()
    for process in processes:
        process.join()

    dis_mat = [return_dic[i] for i in range(n_jobs)]
    dis_mat = np.concatenate(dis_mat, axis=0)
    #print(dis_mat.shape)
    for i in range(dis_mat.shape[0]-1):
        dis_mat[(i+1):,i] = dis_mat[i,(i+1):]

    return dis_mat



def knn_base(pid, index_list, return_dic, dis, k):
    n = len(index_list)
    small_graph = np.zeros([n, dis.shape[1]])
    for i in range(n):
        index = np.argpartition(dis[index_list[i]], k+1)[:(k+1)]
        small_graph[i,index] = 1

    return_dic[pid] = small_graph


def multicore_knn(func, dis, k, n_jobs=-1):
    """parallel k-nearest-neighbor graph"""
    index_list = np.array_split(range(dis.shape[0]), n_jobs)
    processes = []
    return_dic = mp.Manager().dict()

    for i in range(n_jobs):
        proc = mp.Process(target=func, args=(i,index_list[i],return_dic,dis,k))
        processes.append(proc)
        proc.start()
    for process in processes:
        process.join()

    graph = [return_dic[i] for i in range(n_jobs)]
    graph = np.concatenate(graph, axis=0)
    for i in range(graph.shape[0]):
        graph[i,i] = 0

    return graph


def solveW(X, graph, epsilon=1e-4):
    """weight matrix via least square"""
    rank = X.shape[0]
    k = int(sum(graph[0]))
    W = np.zeros([rank, rank])
    for i in range(rank):
        vec = X[i]
        indK = np.where(graph[i]==1)[0]
        mat = X[indK]
        Z = mat - vec
        G = np.dot(Z, Z.T)
        if (epsilon != 0):
            G = G + np.eye(k) * np.trace(G) * epsilon
        w = np.dot(np.linalg.inv(G), np.ones(k))
        w = w/sum(w)
        for ii in range(k):
            W[i, indK[ii]] = w[ii]
        
    return W


def W_base(pid, graph, index_list, X, return_dic, epsilon):
    n = X.shape[0]
    p = len(index_list)
    k = int(sum(graph[0]))
    small_W = np.zeros([p, n])
    for i in range(p):
        ind = index_list[i]
        vec = X[ind]
        indK = np.where(graph[ind]==1)[0]
        mat = X[indK]
        Z = mat - vec
        G = np.dot(Z, Z.T)
        if (epsilon!=0):
            G = G + np.eye(k) * np.trace(G) * epsilon
        w = np.dot(np.linalg.inv(G), np.ones(k))
        w = w/sum(w)
        for ii in range(k):
            small_W[i, indK[ii]] = w[ii]

    return_dic[pid] = small_W


def multicore_W(func, graph, X, epsilon=1e-4, n_jobs=-1):
    """parallel weight matrix via least square"""
    index_list = np.array_split(range(graph.shape[0]), n_jobs)
    processes = []
    return_dic = mp.Manager().dict()

    for i in range(n_jobs):
        proc = mp.Process(target=func, args=(i,graph,index_list[i],X,return_dic,epsilon))
        processes.append(proc)
        proc.start()
    for process in processes:
        process.join()

    W = [return_dic[i] for i in range(n_jobs)]
    W = np.concatenate(W, axis=0)
    
    return W


def eigenFunc(W, n_comp):
    """get the 1:(d+1) smallest eigenvectors"""
    rank = W.shape[0]
    M = np.dot((np.eye(rank)-W).T, np.eye(rank)-W)
    e,v = eigh(M)
   
    return v[:,1:(n_comp+1)]


def LLE(X, k, d, n_jobs=-1, epsilon=1e-4):
    """parallel Locally Linear Embedding"""
    total_cpu = mp.cpu_count()
    if type(n_jobs) is not int or n_jobs < -1 or n_jobs > total_cpu:
        print("Specify appropriate job number!")
        exit(0)
    elif n_jobs==-1:
        n_jobs = total_cpu
    
    n = X.shape[0]
    m = X.shape[1]
    if k > n:
        print("Neighbors more than samples!")
        exit(0)
    if d > m:
        print("Components more than original dimension!")
        exit(0)

    start = time.time()
    dis = multicore_dis(dis_base, X, n_jobs=n_jobs)
    #print("Distance time: %.3fs"%(time.time()-start))
    start = time.time()
    graph = multicore_knn(knn_base, dis, k, n_jobs)
    #print("Graph time: %.3fs"%(time.time()-start))
    start = time.time()
    #W = solveW(X, graph, epsilon=epsilon)
    W = multicore_W(W_base, graph, X, n_jobs=n_jobs, epsilon=epsilon)
    #print("Weight time: %.3fs"%(time.time()-start))
    start = time.time()
    components = eigenFunc(W, d)
    #print("Eigen time: %.3fs"%(time.time()-start))

    return components



if __name__ == "__main__":
    if (len(sys.argv) != 5):
        print("Specify dataName, k and d and n_jobs!")
        exit(0)

    fname = sys.argv[1]
    k = int(sys.argv[2])
    d = int(sys.argv[3])
    n_jobs = int(sys.argv[4])

    data = np.loadtxt("data/"+fname+'.txt', delimiter=',', dtype=float)
    X = data[:,:-1]
    y = data[:,-1]

    start = time.time()
    comp = LLE(X, k, d, n_jobs=n_jobs, epsilon=1e-3)
    print("Time: %.4f s" %(time.time()-start))
    f = plt.figure()
    plt.scatter(comp[:,0], comp[:,1], c=y, s=10, alpha=0.6)
    f.savefig("LLE.pdf")

