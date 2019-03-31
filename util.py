'''
Implemented by Shibo Yao, Mar 31 2019
Utility functions for gather data that are likely manifold distributed
'''
import numpy as np
from scipy import sparse
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer



def get_20news(voc_size = None):
    cate = ['rec.autos', #take a subset only
            'rec.motorcycles',
            'rec.sport.baseball',
            'rec.sport.hockey']
    newsgroups_train = datasets.fetch_20newsgroups(subset='train', categories=cate)
    vectorizer = TfidfVectorizer(max_features=voc_size)
    vectors = vectorizer.fit_transform(newsgroups_train.data)
    labels = newsgroups_train.target
    labels = labels.reshape(-1,1)

    data = sparse.hstack((vectors,labels))
    data = data.tocsr().toarray()

    return data


def read_pgm(pgmf):
    """Return a raster of integers from a PGM as a list of lists."""
    assert pgmf.readline() == b'P5\n'
    (width, height) = [int(i) for i in pgmf.readline().split()]
    depth = int(pgmf.readline())
    assert depth <= 255 

    raster = []
    for y in range(height):
        row = []
        for y in range(width):
            row.append(ord(pgmf.read(1)))
        raster.append(row)

    return raster


def get_ATTface(num_fold = 40, num_img = 10):
    X = []
    y = []
    for i in range(1,num_fold+1):
        for j in range(1,num_img+1):
            fname = 'data/att_faces/s'+str(i)+'/'+str(j)+'.pgm'
            with open(fname, 'rb') as f:
                a = read_pgm(f)
                a = [p for sub in a for p in sub]
                a = [i/255 for i in a]
                X.append(a)
                y.append(i)
    
    X = np.array(X)
    y = np.array(y)
    y = y.reshape(-1,1)
    data = np.hstack((X,y))

    return data


def get_synthetic(name, n_samples = 1000, noise = 0.1):
    if name == 'circles':
        tup = datasets.make_circles(n_samples=n_samples, factor=.5, noise=noise)
    elif name == 'moons':
        tup = datasets.make_moons(n_samples=n_samples, noise=noise)
    elif name == 'roll':
        tup = datasets.make_swiss_roll(n_samples=n_samples, noise=noise)
        y = tup[1]
        c = 5
        mini = min(y)
        step = (max(y) - mini)*1.0001 / c
        steps = [mini+j*step for j in range(c)]
        for i in range(len(y)):
            j = 0
            while j < c and y[i] >= steps[j]:
                j += 1
            y[i] = j
        tup = (tup[0], y)
    else :
        printf("Select circles, moons or roll!")
        exit(0)

    data = np.hstack((tup[0], tup[1].reshape(-1,1)))

    return data


def get_twin(n_samples = 1000, noise = 0.1):
    n = int(n_samples/2)
    r = 1
    angle = np.linspace(0, np.pi*2, n).reshape(-1,1)

    z1 = np.sin(angle)
    y1 = np.cos(angle)
    x1 = np.zeros([n, 1])

    x2 = z1.copy()
    z2 = x1.copy()
    y2 = y1 + 1

    x = np.vstack((x1,x2))
    y = np.vstack((y1,y2))
    z = np.vstack((z1,z2))

    feature = np.hstack((x,y,z))
    feature = feature + np.random.normal(0, noise, feature.shape)
    label = np.vstack((np.zeros([n,1]), np.ones([n,1])))
    data = np.hstack((feature, label))

    return data




if __name__ == "__main__":
    d = get_twin()
    print(d.shape)
    

