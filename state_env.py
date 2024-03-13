# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 15:03:19 2017

environment of the state transfer
reward: 1/(1+e^(30(0.9-x)))
initial state: 100000

@author: pc
"""

import numpy as np
from scipy.linalg import expm,norm
import itertools

STATE_NUM = 6 #the system has 6 spins

MAG = 100  #other parameters
COUPLING = 1
DT = 0.15


nc=3 # number of control sites, nc=3,there are totally 16 actions.
     # nc=4 totally 32 actions. we can only handle nc=3 currently
####################################### defining action, each row of 'mag' corresponds to one allowed configuration of magnetic
def binact(A):                #action label

    m = np.zeros(nc)
    for ii in range(nc):  # transfer action to a binary list, for example: action=5, x=[1, 0, 1, 0], the first and third magnetic is on
        m[nc - 1 - ii] = A >= 2 ** (nc - 1 - ii)
        A = A - 2 ** (nc - 1 - ii) * m[nc - 1 - ii]
    return(m)

mag=[]
for ii in range(8):  #control at the beginning
    mag.append( list( np.concatenate((binact(ii)* MAG,np.zeros(STATE_NUM -nc))) ))

for ii in range(1,8): #control at the end
    mag.append( list( np.concatenate((np.zeros(STATE_NUM -nc),binact(ii)* MAG)) ))

mag.append([MAG for ii in range(STATE_NUM)])
########################################



class State(object):
    def __init__(self):
        super(State, self)
        self.action_space = mag
        self.n_actions = len(self.action_space)  #allowed action number =16
        self.n_features = STATE_NUM*2            #the dimension of input vector

        self.stp=0                              #initially at the first step
        self.stmax=28                            #maximum allowed steps
    
    def reset(self):
        psi = [0 for i in range(STATE_NUM)]  #initial state is [1;0;0;0;0...]
        psi[0] = 1
        self.state = np.array([str(i) for i in psi])
        self.state = np.array(list(itertools.chain(*[(i.real, i.imag) for i in psi])))
        self.stp=0

        return self.state

    def step(self, actionnum): #actionnum: the label of action, n: the step number
        self.stp +=1

        actions = self.action_space[actionnum]  #magnetic field configuration

        ham = np.diag([COUPLING for i in range(STATE_NUM-1)], 1)*(1-0j) + np.diag([COUPLING for i in range(STATE_NUM-1)], -1)*(1+0j) + np.diag(actions)
        statess = [complex(self.state[2*i], self.state[2*i+1]) for i in range(STATE_NUM)]  #transfer real vector to complex vector

        statelist = np.transpose(np.mat(statess))   #to 'matrix'
        next_state = expm(-1j*ham*DT)*statelist  #do operation

        if abs(norm(next_state) - 1.)>1E-8:
            print('FALLO EN LA NORMALIZACION',norm(next_state))

        fidelity = (abs(next_state[-1])**2)[0,0] #calculate fidelity


        ############# reward function
        if fidelity < 0.8:
            reward = fidelity*10
            doned = False
        if fidelity >= 0.8 and fidelity <= 0.95:
            reward = 100/(1+np.exp(10*(0.95-fidelity)))
            doned = False

        doned = False
        if fidelity > 0.95:
            reward = 2500
            doned = True  #break current episode if fidelity>0.95


        reward = reward*(0.95**self.stp)
        ############# a discount is given with respected to step

        next_states = [next_state[i,0] for i in range(STATE_NUM)] #'matrix' to list
        next_states = np.array(list(itertools.chain(*[(i.real, i.imag) for i in next_states]))) #complex to real vector

        self.state = next_states  #this vector is input to the network
        return next_states, reward, doned, fidelity
        
