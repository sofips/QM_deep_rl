import numpy as np
import scipy.linalg as la



def delta(k, n):

    if (k == n):
        d = 1.
    else:
        d = 0.

    return d


def diagonales(bmax, i, nh):
    b = np.full(nh, 0)

    if (i == 1):
        b[0] = 1

    elif (i == 2):

        b[1] = 1

    elif (i == 3):

        b[0] = 1
        b[1] = 1

    elif (i == 4):

        b[2] = 1  # correccion

    elif (i == 5):

        b[0] = 1
        b[2] = 1

    elif (i == 6):

        b[1] = 1
        b[2] = 1

    elif (i == 7):

        b[0] = 1
        b[1] = 1
        b[2] = 1

    elif (i == 8):

        b[nh-3] = 1

    elif (i == 9):

        b[nh-2] = 1

    elif (i == 10):

        b[nh-3] = 1
        b[nh-2] = 1

    elif (i == 11):

        b[nh-1] = 1

    elif (i == 12):

        b[nh-3] = 1
        b[nh-1] = 1

    elif (i == 13):

        b[nh-2] = 1
        b[nh-1] = 1

    elif (i == 14):

        b[nh-3] = 1
        b[nh-2] = 1
        b[nh-1] = 1

    elif (i == 15):

        b[0] = 1
        b[1] = 1
        b[2] = 1
        b[nh-3] = 1
        b[nh-2] = 1
        b[nh-1] = 1

    else:
        b = np.full(nh, 0.)  # correccion

    b = bmax*b

    return b


def acciones(bmax, nh):

    mat_acc = np.zeros((16, nh, nh))

    for i in range(0, 16):

        b = diagonales(bmax, i, nh)

        J =  1 #[-0.5*np.sqrt((nh-k)*k) for k in np.arange(1,nh,1)]

        for k in range(0, nh-1):
            mat_acc[i, k, k+1] = J
            mat_acc[i, k+1, k] = mat_acc[i, k, k+1]

        for k in range(0, nh):

            mat_acc[i, k, k] = b[k]

    return mat_acc
