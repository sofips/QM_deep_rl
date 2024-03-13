'''
Enviroment for spin chains adapted to match X
'''

import numpy as np
from scipy.linalg import expm,norm
import scipy.linalg as la
import itertools
from gym import Env
from gym.spaces import Discrete, Box
import cmath as cm

def diagonales(bmax, i, nh):
    '''
    Define binary actions setting hamiltonian diagonals
    '''
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

    '''
    Generates action matrices
    '''

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
nh = 6
class MyEnv(Env):

    def __init__(self):  # esto inicializa el ambiente
        
        self.bm = 100
        self.env_features = 2*nh
        actions = acciones(self.bm, nh)

        self.action_space = actions  # 16 acciones posibles
        self.observation_space = Box(low=np.zeros(2*nh),
                                     high=np.ones(2*nh))
        self.n = nh
        self.n_actions = 16  #allowed action number =16
        self.n_features = 2*nh  

        # valor del campo magnetico
        comp_i = complex(0, 1)
        self.en = np.zeros((16, nh), dtype=np.complex_)
        self.bases = np.zeros((16, nh, nh), dtype=np.complex_)
        self.propagadores = np.zeros((16, nh, nh), dtype=np.complex_)
        self.desc_esp = np.zeros((16, nh, nh), dtype=np.complex_)

        self.t = 0.                          # inicializo el tiempo en 0
        self.dt = 0.15                      # intervalos de tiempo
        self.tol = 0.05                      # tolerancia
        self.tmax = 28                      # tiempo maximo


        for j in range(0, 16): # para cada matriz de accion

                        self.en[j, :], self.bases[j, :, :] = la.eig(actions[j, :, :])

                        for k in range(0, nh):
                            p = np.outer(self.bases[j, :, k], self.bases[j, :,k])

                            self.propagadores[j, :, :] = (
                                self.propagadores[j, :, :]
                                + cm.exp(-comp_i * self.dt * self.en[j, k]) * p
                            )

                            self.desc_esp[j,:,:] = self.desc_esp[j,:,:] + p * self.en[j, k]


        # check de descomposición espectral
        check_de = True

        for k in np.arange(0,16):
                for i in np.arange(0,nh):
                        for j in np.arange(0,nh):
        
                            if actions[k,i,j]-self.desc_esp[k,i,j] > 1E-8:
                                    print('error desc. esp')
                                    check_de = False
        
        if check_de:
             print('Descomposicion espectral: correcta')
                            
        check_prop = True

        for a in np.arange(0,16):
            for j in np.arange(0,nh):
                    errores = np.matmul(self.propagadores[a,:,:],self.bases[a,:,j]) - np.exp(-comp_i*self.dt*self.en[a,j])*self.bases[a,:,j] 
                    et = np.sum(errores)
                    if la.norm(et)>1E-8:
                         print('error en propagacion')
                         check_prop = False
                         
        if check_prop:
             print('Propagacion de autoestados: correcta')

    def step(self, actionnum):
        
        self.stp +=1
        
        self.old_state = np.transpose(self.old_state)

        self.cstate = [complex(self.state[2*i], self.state[2*i+1]) for i in range(nh)]  #transfer real vector to complex vector #np.zeros(nh, dtype=np.complex_)
        self.cstate = np.transpose(np.mat(self.cstate))
        self.t = self.stp #self.dt*self.stp

        self.old_cstate = [complex(self.old_state[2*i], self.old_state[2*i+1]) for i in range(nh)]  #transfer real vector to complex vector #np.zeros(nh, dtype=np.complex_)
        self.old_cstate = np.transpose(np.mat(self.old_cstate))


        
        j = 0

        for i in np.arange(0,nh):
            self.old_cstate[i] = complex(self.old_state[j],self.old_state[j+1])
            j+=2
        
        self.cstate = np.matmul(self.propagadores[actionnum, :, :], self.cstate)
        self.old_cstate = np.matmul(self.propagadores[actionnum, :, :], self.old_cstate)



        for i in np.arange(0,2*nh,2):
            self.old_state[i] = np.real(self.old_cstate[i//2])
            self.old_state[i+1] = np.imag(self.old_cstate[i//2])


        fid = np.real(self.cstate[nh-1]*np.conjugate(self.cstate[nh-1]))[0,0]
        old_fid = np.real(self.old_cstate[nh-1]*np.conjugate(self.old_cstate[nh-1]))[0,0]



        if (old_fid <= 0.8):
            old_reward = 10*old_fid
        elif (0.8 <= old_fid <= 1 - self.tol):
            old_reward = 100 / (1 + np.exp(10 * (1 - self.tol - old_fid)))
        else:
            old_reward = 2500

        if (old_fid >= 1 - self.tol) or (self.t >= self.tmax):
            done = True
        else:
            done = False

        if (fid <= 0.8):
            reward = 10*fid
        elif (0.8 <= fid <= 1 - self.tol):
            reward = 100 / (1 + np.exp(10 * (1 - self.tol - fid)))
        else:
            reward = 2500

        if (fid >= 1 - self.tol) or (self.t >= self.tmax):
            done = True
        else:
            done = False

        if abs(norm(self.old_cstate) - 1.)>1E-8:
            print('FALLO EN LA NORMALIZACION',norm(self.cstate))

        reward = reward*(0.95**self.stp)
        old_reward = old_reward*(0.95**self.stp)

        self.old_state = np.transpose(self.old_state)

        self.cstate = [ self.cstate [i,0] for i in range(nh)] #'matrix' to list
        self.state = np.array(list(itertools.chain(*[(i.real, i.imag) for i in self.cstate]))) #complex to real vector

        if (norm(self.old_state - self.state) > 1E-8):
            print('estado')
            stop
        if (reward != old_reward):
            print('recompensa')
            stop
        if (fid != old_fid):
            print('fidelidad')
            stop
        #print('Antes: ', (self.old_state), old_fid)
        #print('Despues: ', (self.state), fid)
        estado = self.old_state
        
   
        return estado, old_reward, done, old_fid

    def reset(self):  # se resetean el tiempo y el estado
        
        psi = [0 for i in range(nh)]  #initial state is [1;0;0;0;0...]
        psi[0] = 1
        #self.state = np.array([str(i) for i in psi])
        self.state = np.array(list(itertools.chain(*[(i.real, i.imag) for i in psi])))
        
        self.old_state = np.zeros(2*self.n, dtype=np.float_)
        self.old_state[0] = 1

        self.t = 0.
        self.stp = 0

        return self.old_state
        #return self.state