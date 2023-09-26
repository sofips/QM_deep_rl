import numpy as np
import scipy.linalg as la
import cmath as cm
import numpy as np
import scipy.linalg as la



def delta(k, n):

    if (k == n):
        d = 1.
    else:
        d = 0.

    return d


def diagonales(bmax, i, nh):

    if (i == 1):
        b = np.full(nh, -1)
        b[0] = 1

    elif (i == 2):
        b = np.full(nh, -1)

        b[1] = 1

    elif (i == 3):

        b = np.full(nh, -2.)

        b[0] = 0.
        b[1] = 0.

    elif (i == 4):

        b = np.full(nh, -1)

        b[2] = 1  # correccion

    elif (i == 5):

        b = np.full(nh, -2.)

        b[0] = 0.
        b[2] = 0.

    elif (i == 6):

        b = np.full(nh, -2.)  # correccion

        b[1] = 0.
        b[2] = 0.

    elif (i == 7):

        b = np.full(nh, -3)

        b[0] = -1
        b[1] = -1
        b[2] = -1

    elif (i == 8):
        b = np.full(nh, -1)
        b[nh-3] = 1

    elif (i == 9):

        b = np.full(nh, -1)
        b[nh-2] = 1

    elif (i == 10):

        b = np.full(nh, -2.)
        b[nh-3] = 0.
        b[nh-2] = 0.

    elif (i == 11):

        b = np.full(nh, -1)
        b[nh-1] = 1

    elif (i == 12):
        b = np.full(nh, -2.)

        b[nh-3] = 0.
        b[nh-1] = 0.

    elif (i == 13):

        b = np.full(nh, -2.)

        b[nh-2] = 0.  # correccion
        b[nh-1] = 0.  # correccion

    elif (i == 14):

        b = np.full(nh, -3)

        b[nh-3] = -1
        b[nh-2] = -1
        b[nh-1] = -1

    elif (i == 15):

        b = np.full(nh, -2.)
    else:
        b = np.full(nh, 0.)  # correccion

    b = bmax*b

    return b


def actions(bmax, nh):

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


def gen_props(actions,n,b,dt):

        mat_acc = actions
        comp_i = complex(0, 1)
        en = np.zeros((16, n), dtype=np.complex_)
        bases = np.zeros((16, n, n), dtype=np.complex_)
        propagadores = np.zeros((16, n, n), dtype=np.complex_)
        desc_esp = np.zeros((16, n, n), dtype=np.complex_)


        for j in range(0, 16): # para cada matriz de accion

                        en[j, :], bases[j, :, :] = la.eig(mat_acc[j, :, :])

                        for k in range(0, n):
                            p = np.outer(bases[j, :, k], bases[j, :,k])

                            propagadores[j, :, :] = (
                                propagadores[j, :, :]
                                + cm.exp(-comp_i * dt * en[j, k]) * p
                            )

                            desc_esp[j,:,:] = desc_esp[j,:,:] + p * en[j, k]


        # check de descomposición espectral
        check_de = True

        for k in np.arange(0,16):
                for i in np.arange(0,n):
                        for j in np.arange(0,n):
        
                            if mat_acc[k,i,j]-desc_esp[k,i,j] > 1E-8:
                                    print('error desc. esp')
                                    check_de = False
        
        if check_de:
             print('Descomposicion espectral: correcta')
                            
        check_prop = True

        for a in np.arange(0,16):
            for j in np.arange(0,n):
                    errores = np.matmul(propagadores[a,:,:],bases[a,:,j]) - np.exp(-comp_i*dt*en[a,j])*bases[a,:,j] 
                    et = np.sum(errores)
                    if la.norm(et)>1E-8:
                         print('error en propagacion')
                         check_prop = False
                         
        if check_prop:
             print('Propagacion de autoestados: correcta')

        return propagadores

def fidelity(action_sequence,props):
    n = np.shape(props)[1]
    state = np.zeros(n, dtype=np.complex_)
    state[0] = 1.
    max_fid = 0.

    for action in action_sequence:
    
        state = np.matmul(props[action, :, :], state)
        fid = np.real(state[n-1]*np.conjugate(state[n-1]))

        if fid>max_fid:
             max_fid = fid


    if abs(la.norm(state) - 1.)>1E8:

        print('FALLO EN LA NORMALIZACION',la.norm(state))

    return max_fid

def fitness_func_constructor(fid_function,arguments):
    '''
    Parameters:
        - fidelity function(can be either fidelity or en_fidelity)
        - arguments: arguments of fidelity functions
    Return:
        - lambda function: the fitness function as required by PyGAD
    '''
    fitness = lambda vec: fid_function(vec,*arguments)

    return lambda ga_instance, solution, solution_idx: fitness(solution)

def generation_func(ga):
    
    solution, solution_fitness, solution_idx = ga.best_solution()
    
    print('Generation', ga.generations_completed)
    print('Solution: ', solution, 'Fitness: ', solution_fitness)
