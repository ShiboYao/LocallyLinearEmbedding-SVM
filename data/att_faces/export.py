import numpy as np


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



if __name__ == "__main__":
    num_fold = 40
    num_img = 10
    X = []
    y = []

    for i in range(1,num_fold+1):
        for j in range(1,num_img+1):
            fname = 's'+str(i)+'/'+str(j)+'.pgm'
            with open(fname, 'rb') as f:
                a = read_pgm(f)
                a = [p for sub in a for p in sub]
                a = [i/255 for i in a]
                X.append(a)
                y.append(i)
    
    X = np.array(X)
    y = np.array(y)
    y = y.reshape(-1,1)
    print(X.shape, y.shape)
    data = np.hstack((X,y))
    np.savetxt('../att_face.txt', data, fmt='%.8e', delimiter=',', newline='\n')
