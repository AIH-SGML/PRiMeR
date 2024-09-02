import numpy as np
import scipy.stats as st


def toRanks(A):
    AA = np.zeros_like(A)
    for i in range(A.shape[1]):
        AA[:, i] = st.rankdata(A[:, i])
    AA = np.array(np.around(AA), dtype="int") - 1
    return AA


def gaussianize(Y):
    N, P = Y.shape

    YY = toRanks(Y)
    quantiles = (np.arange(N) + 0.5) / N
    gauss = st.norm.isf(quantiles)
    Y_gauss = np.zeros((N, P))
    for i in range(P):
        Y_gauss[:, i] = gauss[YY[:, i]]
    Y_gauss *= -1
    return Y_gauss
