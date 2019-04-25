import numpy as np

def logist(x, k, max, min, mean):
    """
    For calculating the decay factor df.
    Higher grad = sharper
    """
    max = max - min;
    y = min + (max / (1 + np.exp(k * (x - mean))))
    return y

def spm_norm(A):
    """
    Normalisation of a probability transition matrix (columns)
    """
    A = np.dot(A, np.diag(1 / np.sum(A, 0)))
    return A

def spm_softmax(x, k = 1):
    """
    Softmax (neural transfer) function of COLUMN vectors
    Format [y] = spm_softmax(x,k)
    
    x - vector of activity
    k - temperature or inverse sensitivity parameter (default k = 1)
    
    y = exp(k*x)/sum(exp(k*x))
    
    NB: If supplied with a matrix this rotine will return the softmax
    function over columns - so that spm_softmax([x1,x2,...]) = [1,1,...]
    """
    x = x * k
    
    x = x - np.max(x)
    ex = np.exp(x)
    y = ex / np.sum(ex, axis = 0)
    
    return y