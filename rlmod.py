import numpy as np
import scipy.linalg as la


def gen_base(nh):

    H = np.full((nh, nh), 0.)
    J = -0.5

    for i in range(0, nh-1):
        H[i, i+1] = J
        H[i+1, i] = H[i, i+1]

    e, base = la.eig(H)

    return e, base


def delta(k, n):

    if (k == n):
        d = 1.
    else:
        d = 0.

    return d


def diagonales(bmax, i, nh):

    if (i == 1):
        b = np.full(nh, -0.5)
        b[0] = 0.5

    elif (i == 2):
        b = np.full(nh, -0.5)

        b[1] = 0.5

    elif (i == 3):

        b = np.full(nh, -1.)

        b[0] = 0.
        b[1] = 0.

    elif (i == 4):

        b = np.full(nh, -0.5)

        b[2] = 0.5  # correccion

    elif (i == 5):

        b = np.full(nh, -1.)

        b[0] = 0.
        b[2] = 0.

    elif (i == 6):

        b = np.full(nh, -1.)  # correccion

        b[1] = 0.
        b[2] = 0.

    elif (i == 7):

        b = np.full(nh, -1.5)

        b[0] = -0.5
        b[1] = -0.5
        b[2] = -0.5

    elif (i == 8):
        b = np.full(nh, -0.5)
        b[nh-3] = 0.5

    elif (i == 9):

        b = np.full(nh, -0.5)
        b[nh-2] = 0.5

    elif (i == 10):

        b = np.full(nh, -1.)
        b[nh-3] = 0.
        b[nh-2] = 0.

    elif (i == 11):

        b = np.full(nh, -0.5)
        b[nh-1] = 0.5

    elif (i == 12):
        b = np.full(nh, -1.)

        b[nh-3] = 0.
        b[nh-1] = 0.

    elif (i == 13):

        b = np.full(nh, -1.)

        b[nh-2] = 0.  # correccion
        b[nh-1] = 0.  # correccion

    elif (i == 14):

        b = np.full(nh, -1.5)

        b[nh-3] = -0.5
        b[nh-2] = -0.5
        b[nh-1] = -0.5

    elif (i == 15):

        b = np.full(nh, -2.)
    else:
        b = np.full(nh, 0.)  # correccion

    b = bmax*b

    return b


def acciones(bmax, nh):

    mat_acc = np.zeros((16, nh, nh))

    for i in range(0, 16):

        b = diagonales(bmax, i, nh)

        J =  -0.5 #[-0.5*np.sqrt((nh-k)*k) for k in np.arange(1,nh,1)]

        for k in range(0, nh-1):
            mat_acc[i, k, k+1] = J
            mat_acc[i, k+1, k] = mat_acc[i, k, k+1]

        for k in range(0, nh):

            mat_acc[i, k, k] = b[k]

    return mat_acc
