import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import rlmod as rk
import scipy.linalg as la
import csv
import cmath as cm


class MyEnv(Env):

    def __init__(self, nh):  # esto inicializa el ambiente

        self.action_space = Discrete(16)  # 16 acciones posibles
        self.observation_space = Box(low=np.zeros(nh, dtype=np.complex_),
                                     high=np.ones(nh, dtype=np.complex_))
        self.n = nh
        # valor del campo magnetico
        self.bm = 100
        self.mat_acc = rk.acciones(self.bm, nh)
        comp_i = complex(0, 1)
        self.en = np.zeros((16, nh), dtype=np.complex_)
        self.bases = np.zeros((16, nh, nh), dtype=np.complex_)
        self.propagadores = np.zeros((16, nh, nh), dtype=np.complex_)
        self.desc_esp = np.zeros((16, nh, nh), dtype=np.complex_)

        self.t = 0.                          # inicializo el tiempo en 0
        self.dt = 0.15                      # intervalos de tiempo
        self.tol = 0.05                      # tolerancia
        self.tmax = 32                      # tiempo maximo


        for j in range(0, 16): # para cada matriz de accion

                        self.en[j, :], self.bases[j, :, :] = la.eig(self.mat_acc[j, :, :])

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
        
                            if self.mat_acc[k,i,j]-self.desc_esp[k,i,j] > 1E-8:
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

        c0 = np.zeros(nh, dtype=np.complex_)
        c0[0] = 1.

    def step(self, action):

        self.t = self.t + self.dt
        self.state = np.matmul(self.propagadores[action, :, :], self.state)

        fid = np.real(self.state[nh-1]*np.conjugate(self.state[nh-1]))
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

        if abs(la.norm(self.state) - 1.)>1E8:
            print('FALLO EN LA NORMALIZACION',la.norm(self.state))

        info = {}

        return self.state, reward, done, info

    def reset(self):  # se resetean el tiempo y el estado

        c0 = np.zeros(self.n)
        c0[0] = 1.
        self.state = c0
        self.t = 0.
        return self.state


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        # store one hot encoding of actions, if appropriate
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal



class DeepQNetwork(nn.Module):
    def __init__(self, lr, nh, fc1_dims, fc2_dims,
                 n_actions):
        super(DeepQNetwork, self).__init__()
        self.nh = nh
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.nh, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state.float()))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions

class Agent(object):

    def __init__(self, gamma, epsilon, lr, nh, batch_size, n_actions,
                 max_mem_size=40000, eps_end=0.01, eps_dec=0.0001):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 200

        self.Q_eval = DeepQNetwork(lr, 
                                   nh=nh,
                                   fc1_dims=120, fc2_dims=120, n_actions=n_actions)
        self.state_memory = np.zeros((self.mem_size, *nh),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *nh),
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype= bool)

    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(
                self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(
                self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(
                self.terminal_memory[batch]).to(self.Q_eval.device)

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma*T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.iter_cntr += 1
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min


# --------------------------------------------------------------------------
if __name__ == '__main__':

    nh = 7
    env = MyEnv(nh)
    states = env.observation_space.shape
    actions = env.action_space.n

    lr = 0.01
    n_games = 50000
  
    agent =  Agent(gamma=0.95, epsilon=1.0, batch_size=32, n_actions=16, eps_end=0.01,
                  nh=[nh], lr=lr)

    scores = []
    fids = []
    tmaxs = []

    eps_history = []

    dt = 0.15
    f1 = open(sys.argv[1], "w")
    writer = csv.writer(f1)
    
    for i in range(n_games):

        done = False
        score = 0
        observation = env.reset()
        fid0 = 0.
        t = 0.
        indt = 0
        tfmax = 0.

        #if (i % 1000 == 0):
         #   fname = 'fid_eps_' + str(i) + '.dat'
          #  f2 = open(fname, "w")
           # writer2 = csv.writer(f2)

        while not done:
            indt += 1
            t = indt*dt
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += np.real(reward)
            agent.store_transition(observation, action, reward, 
                                    observation_, done)
            observation = observation_
            
            if (indt % 32 == 0):
                agent.learn()

            fid = np.real(observation[nh-1]*np.conjugate(observation[nh-1]))

            if (fid > fid0):
                fid0 = np.real(fid)
                tfmax = t

            #if (i % 1000 == 0):
                #c02 = np.real(observation[0]*np.conjugate(observation[0]))
                #row2 = [t, c02, fid]
                #writer2.writerow(row2)

        #if (i % 1000 == 0):
            #f2.close()

        eps_history.append(agent.epsilon)
        scores.append(score)
        tmaxs.append(tfmax)
        fids.append(fid0)

        avg_score = np.mean(scores[max(0, i-100):(i+1)])
        avg_fids = np.mean(fids[max(0, i-100):(i+1)])
        avg_tf = np.mean(tmaxs[max(0, i-100):(i+1)])

        print('episode: ', i, 'score: %.2f' % score,
              'average score %.2f' % avg_score, 'fidelidad: %.2f' % fid0, 
              'fid. media: %.2f' %avg_fids, 'epsilon: %.2f' %agent.epsilon)

        row = [i, np.real(fid0), np.real(tfmax), np.real(score),
               np.real(avg_fids), np.real(avg_tf), np.real(avg_score),
               np.real(agent.epsilon)
               ]
        writer.writerow(row)

        #if i % 100 == 0 and i > 0:
         #   agent.save_model()
        
    f1.close()


