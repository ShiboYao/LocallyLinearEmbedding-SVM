'''
Implemented by Shibo Yao, Mar 31 2019
Visualization of linear dimensdion reduction and 
non-linear dimension reduction on manifold distributed data
'''
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from util import *
from spectral import LLE
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D



if __name__ == "__main__":
    n_samples = 1000
    noise = 0.1

    moons = get_synthetic('moons', n_samples, noise)
    #circles = get_synthetic('circles', n_samples, noise)
    #roll = get_synthetic('roll', n_samples, noise)
    sphere = np.loadtxt('data/ionosphere.txt', delimiter=',', dtype=float)
    digits = np.loadtxt('data/digits.txt', delimiter=',', dtype=float)
    #news = get_20news(voc_size = 10000)
    #attface = get_ATTface(num_fold=40, num_img=10)
    twin = get_twin(n_samples, noise)

    data_dic = {'twin':twin,
                'moons':moons,
                #'circles':circles,
                #'roll':roll,
                'sphere':sphere,
                'digits':digits
                #'attface':attface,
                #'news':news
                }

    k = 10
    d = 2
    
    for key in data_dic:
        print(key)
        data = data_dic[key]
        X = data[:,:-1]
        y = data[:,-1]
        
        fig = plt.figure()
        plt.suptitle(key)

        comp_LLE = LLE(X, k, d, n_jobs=-1, epsilon=1e-3)
        plt.subplot(1,2,1)
        plt.title('LLE')
        plt.scatter(comp_LLE[:,0], comp_LLE[:,1], c=y, s=10, alpha=0.6)
        comp_PCA = PCA(n_components=d).fit_transform(X)
        plt.subplot(1,2,2)
        plt.title('PCA')
        plt.scatter(comp_PCA[:,0], comp_PCA[:,1], c=y, s=10, alpha=0.6)

        fig.savefig(key+'.pdf')
        print('\n')
    
    fig3d = plt.figure()
    ax = fig3d.add_subplot(111, projection='3d')
    ax.scatter(twin[:,1], twin[:,0], twin[:,2], c=twin[:,3])
    plt.savefig('twin3d.pdf')
