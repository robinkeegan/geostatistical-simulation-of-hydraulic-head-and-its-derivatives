import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
# covariance function and derivatives
def rbf(X1, X2, l, sigma):
    h = cdist(X1, X2)
    return sigma * np.exp(-h ** 2 / (2 * l ** 2))

def d1rbf(X1, X2, l, sigma, axis=0):
    d = X1[:, axis].reshape(-1, 1) - X2[:, axis].reshape(1, -1)
    return (d * rbf(X1, X2, l, sigma)) / l ** 2

def d2rbf(X1, X2, l, sigma, axis=0):
    d = X1[:, axis].reshape(-1, 1) - X2[:, axis].reshape(1, -1)
    K = rbf(X1, X2, l, sigma)
    return (d ** 2 * K) / l ** 4 - K / l ** 2

def dxyrbf(X1, X2, l, sigma, axis1=0, axis2=1):
    d1 = (X1[:, axis1].reshape(-1, 1) - X2[:, axis1].reshape(1, -1))
    d2 = (X1[:, axis2].reshape(-1, 1) - X2[:, axis2].reshape(1, -1))
    K = rbf(X1, X2, l, sigma)
    return (d1 * d2 * K) / l ** 4 - K / l ** 3
# Best linear unbaised estimate of the mean
def global_mean(K, y):
    n = np.size(y)
    mw = np.linalg.solve(K, np.ones(n))
    gmu = mw.T.dot(y) / mw.T.dot(np.ones(n))
    return gmu
# Kriging functions
def krige(Kii, Kij, Kjj, y, gmu=0):
    w = np.linalg.solve(Kii, Kij)
    mu = w.T.dot(y) + (1 - w.T.dot(np.ones(y.size))) * gmu
    cov = Kjj - w.T.dot(Kij)
    return mu, cov

def krige_gradient(Kii, Kij, y):
    w = np.linalg.solve(Kii, Kij)
    mu = w.T.dot(y) #assuming a constant mean the derivative of the mean is zero thus the last term is dropped
    return mu

# functions for generating realisations of function values and derivatives
def find_closest(new, obs, k=2):
    n = np.size(obs, axis=0)
    if n <= k:
        return np.arange(n)
    else:
        d = cdist(obs, new)[:, 0]
        return np.argpartition(d, k)[:k]

def FunctionRealisation(X, Xnew, y, p, nreal=1, alpha=1e-6, k=100):
    n, m = np.size(X, axis=0), np.size(Xnew, axis=0)
    realisations = np.zeros((nreal, m))
    ind = np.arange(m)
    gmu = global_mean(rbf(X, X, *p) + np.eye(y.size) * alpha, y)
    for j in tqdm(range(nreal)):
        f, fnew, Xobs = y, np.zeros(m)*np.nan, X
        alpha_ = np.ones(n) * alpha
        np.random.shuffle(ind)
        for i in range(m):
            indx = find_closest(Xnew[ind][i:i+1], Xobs, k)
            Kii = rbf(Xobs[indx], Xobs[indx], *p) + np.eye(indx.size) * alpha_[indx]
            Kij = rbf(Xobs[indx], Xnew[ind][i:i+1], *p)
            Kjj = rbf(Xnew[ind][i:i+1], Xnew[ind][i:i+1], *p)
            mu, cov = krige(Kii, Kij, Kjj, f[indx], gmu)
            std = np.where(cov>0, np.sqrt(cov.item()), 0)
            fnew[i] = np.random.normal(mu.item(), std)
            f = np.concatenate((f, fnew[i:i+1]))
            Xobs = np.concatenate((Xobs, Xnew[ind][i:i+1]))
            alpha_ = np.concatenate((alpha_, [1e-8]))
        ind = np.argsort(ind)
        realisations[j] = fnew[ind]
    return realisations

def DerivativeRealisation(X, Xnew, y, yr, p, nreal=1, alpha=1e-6, k=100):
    n, m = np.size(X, axis=0), np.size(Xnew, axis=0)
    realisations = np.zeros((nreal, m))
    dx, dy, dxx, dyy, dxy = np.zeros((5, nreal, m))
    Xobs = np.concatenate((X, Xnew))
    alpha_ = np.concatenate((alpha * np.ones(n), np.ones(m) * 1e-8))
    for j in tqdm(range(nreal)):
        f = np.concatenate((y, yr[j]))
        for i in range(m):
            indx = find_closest(Xnew[i:i+1], Xobs, k)
            Kii = rbf(Xobs[indx], Xobs[indx], *p) + np.eye(indx.size) * alpha_[indx]
            dx[j, i] = krige_gradient(Kii, d1rbf(Xobs[indx], Xnew[i:i+1], *p, 0), f[indx])
            dy[j, i] = krige_gradient(Kii, d1rbf(Xobs[indx], Xnew[i:i+1], *p, 1), f[indx])
            dxx[j, i] = krige_gradient(Kii, d2rbf(Xobs[indx], Xnew[i:i+1], *p, 0), f[indx])
            dyy[j, i] = krige_gradient(Kii, d2rbf(Xobs[indx], Xnew[i:i+1], *p, 1), f[indx])
            dxy[j, i] = krige_gradient(Kii, dxyrbf(Xobs[indx], Xnew[i:i+1], *p, 0, 1), f[indx])
    return dx, dy, dxx, dyy, dxy
# function to conduct the second derivative test
def SecondDerivativeTest(dxx, dyy, dxy):
    classification = np.zeros(dxx.shape)
    H = dxx * dyy - dxy ** 2
    classification[(H < 0)] = 1 #saddle point
    classification[(H > 0) & (dxx < 0)] = 2 #maxmimum
    classification[(H > 0) & (dxx > 0)] = 3 #minimum
    classification[(H == 0)] = 4 #unknown
    return H, classification
