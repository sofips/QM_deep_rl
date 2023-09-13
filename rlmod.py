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


def mat_mag(nh, base):

    sz = np.full((nh, nh, nh), 0.)

    for k in range(0, nh):
        for i in range(0, nh):
            for j in range(0, nh):
                for m in range(0, nh):

                    sz[k, i, j] = sz[k, i, j] + base[i, m] * \
                        base[j, m]*(delta(k, nh-1)-0.5)

    return sz


def delta(k, n):

    if (k == n):
        d = 1.
    else:
        d = 0.

    return d


def rk4_step(nh, c0, dt, t, e, base, b, sz):

    k1 = np.zeros(nh)
    k2 = np.zeros(nh)
    k3 = np.zeros(nh)
    k4 = np.zeros(nh)
    c = np.zeros(nh)

    k1 = dt*f(nh, t, c0, e, b, sz)

    k2 = dt*f(nh, t+dt*0.5, c0+k1/2., e, b, sz)

    k3 = dt*f(nh, t+dt/2*0.5, c0+k2/2., e, b, sz)

    k4 = dt*f(nh, t+dt, c0+k3, e, b, sz)

    for i in range(0, nh):
        c[i] = c0[i] + (k1[i] + 2.*k2[i] + 2. * k3[i] + k4[i])/6

    return c


def f(nh, t, c0, e, b, sz):

    ff = np.zeros(nh)

    comp_i = complex(0, 1)

    for i in range(0, nh):

        sumj = 0.

        for j in range(0, nh):

            sumk = 0.

            for k in range(0, nh):
                sumk = sumk + b[k]*sz[k, i, j]

            sumj = sumj + c0[j]*np.exp(-comp_i*(e[j]-e[i])*t)*sumk

        ff[i] = -comp_i*sumj

    return ff


def modpsi(nh, c):

    mp = 0.

    for i in range(0, nh):
        mp = mp + c[i]*np.conjugate(c[i])

    return mp


def fidelidad(nh, e, base, c, t):

    fidc = 0.
    comp_i = complex(0, 1)

    for i in range(0, nh-1):
        fidc = fidc + c[i]*np.exp(-comp_i*e[i]*t)*base[i, nh-1]

    fid = np.real(fidc)*np.real(fidc)+np.imag(fidc)*np.imag(fidc)

    return fid


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

        b[2] = 0.5 # correccion

    elif (i == 5):

        b = np.full(nh, -1.)

        b[0] = 0.
        b[2] = 0.

    elif (i == 6):

        b = np.full(nh, -1.) #correccion

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

        b[nh-2] = 0. #correccion
        b[nh-1] = 0. #correccion

    elif (i == 14):

        b = np.full(nh, -1.5)

        b[nh-3] = -0.5
        b[nh-2] = -0.5
        b[nh-1] = -0.5

    elif (i == 15):

        b = np.full(nh, -2.)
    else:
        b = np.full(nh, 0.) # correccion

    b = bmax*b

    return b


def acciones(bmax, nh):

    mat_acc = np.zeros((16, nh, nh))

    for i in range(0, 16):

        b = diagonales(bmax, i, nh)

        J = -0.5

        for k in range(0, nh-1):
            mat_acc[i, k, k+1] = J
            mat_acc[i, k+1, k] = mat_acc[i, k, k+1]

        for k in range(0, nh):

            mat_acc[i, k, k] = b[k]

    return mat_acc
