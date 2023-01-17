'''
Código utilizado para implementar el algoritmo de RL desarrollado en
el trabajo de Zhang et. al (https://doi.org/10.1103/PhysRevA.97.052333).
El mismo permite obtener secuencias óptimas de campos magnéticos a
aplicarse sobre los extremos de una cadena de spin para transmitir
el estado correspondiente al primer elemento de la misma hasta el
último //

Implementation of the algorithm developed in the work of Zhang et.
al (https://doi.org/10.1103/PhysRevA.97.052333). The following
code uses Deep Reinforcement Learning to obtain an optimal
sequence of magnetic fields that should be applied to the
extremes of a spin chain in order to achieve a perfect
transmission.
'''

from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import rkmod as rk
import scipy.linalg as la
import csv
import cmath as cm


class MyEnv(Env):

    '''
    The enviroment class defines:
        - 16 possible actions corresponding to different combinations of
          "on" and "off" magnetic fields (see referenced work) and the 16
          associated evolution matrices necessary to aply said actions to the
          system.
        - Characteristics of the enviroment such as magnetic field
         strength (bm),length of the spin chain (nh), max. time
         and length/quantity of time steps.
        - The state of the system which is specified using the normalized
          coefficients corresponding to the "one-excitation" basis. The set
          of every possible state is the observation_space
    '''

    def __init__(self, nh):

        self.action_space = Discrete(16)  # 16 possible actions

        self.observation_space = Box(low=np.zeros(nh, dtype=np.complex_),
                                     high=np.ones(nh, dtype=np.complex_))

        self.n = nh  # number of spins
        self.bm = 100  # magnetic field strength

        self.mat_acc = \
            rk.acciones(self.bm, nh)  # Hamiltonian of every action

        comp_i = complex(0, 1)  # complex unit i

        self.en = \
            np.zeros((16, nh), dtype=np.complex_)  # Storage of eigenvalues

        self.bases = \
            np.zeros((16, nh, nh), dtype=np.complex_)  # " of eigenvectors

        self.propagadores = np.zeros(
            (16, nh, nh), dtype=np.complex_)  # evolution operators

        self.t = 0.                          # set time to 0
        self.dt = 0.15                       # length of time intervals
        self.tol = 0.05                      # tolerance in transm. fidelity
        self.tmax = 32                       # max time of evolution

        # evolution operators for each action are calculated using energy basis

        for j in range(0, 16):

            self.en[j, :], self.bases[j, :, :] = la.eig(self.mat_acc[j, :, :])

            for k in range(0, nh):
                p = np.outer(self.bases[j, k, :], self.bases[j, k, :])

                self.propagadores[j, :, :] = (
                    self.propagadores[j, :, :] +
                    cm.exp(-comp_i * self.dt * self.en[j, k]) * p
                )

        # initial state is set to the one with only the first spin up

        c0 = np.zeros(nh, dtype=np.complex_)
        self.e0, self.base0 = rk.gen_base(nh)
        c0[0] = 1.

    def step(self, action):
        '''
        Describes what happens in each time step, not to be
        confused with full episodes (evolution from t=0 to tmax) or
        learning stages.
        '''

        self.t = self.t + self.dt   # time increases in "dt"

        # state updated with the corresponding evolution operator

        self.state = \
            np.matmul(self.propagadores[action, :, :], self.state)

        # fidelity of transmission is calculated

        fid = np.real(self.state[nh-1]*np.conjugate(self.state[nh-1]))

        # instant reward as defined in the work of reference

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

        info = {}

        return self.state, reward, done, info

    def reset(self):
        '''
        Resets the system to initial state, setting time to cero
        and the state to the one with only the first spin up
        '''

        c0 = np.zeros(self.n)
        c0[0] = 1.
        self.state = c0
        self.t = 0.
        return self.state


'''
The following classes (Replay Buffer and Agent) along with the
construction of the neural network were based in the codes and
tutorials from Phil Tabor which can be found at:
https://github.com/philtabor/Youtube-Code-Repository/
tree/master/ReinforcementLearning/DeepQLearning
'''


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


def build_dqn(lr, n_actions, nh, fc1_dims, fc2_dims):
    model = Sequential([
        Dense(fc1_dims, input_shape=(nh,)),
        Activation('relu'),
        Dense(fc2_dims),
        Activation('relu'),
        Dense(n_actions)])
    model.compile(optimizer=Adam(lr=lr), loss='mse')

    return model


class Agent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon,
                 batch_size, nh, epsilon_dec=0.999999,
                 epsilon_end=0, mem_size=1000000, fname='dqn_model.h5'):

        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBuffer(mem_size, nh, n_actions,
                                   discrete=True)
        self.q_eval = build_dqn(alpha, n_actions, nh, 120, 120)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):

        state = state[np.newaxis, :]
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.q_eval.predict(state, verbose=0)
            action = np.argmax(actions)

        return action

    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)

            q_eval = self.q_eval.predict(state, verbose=0)

            q_next = self.q_eval.predict(new_state, verbose=0)

            q_target = q_eval.copy()

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = \
                reward + self.gamma*np.max(q_next, axis=1)*done

            _ = self.q_eval.fit(state, q_target, verbose=0)

            self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > \
                self.epsilon_min else self.epsilon_min

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)


'''
Main program, implements the algorithm described in
referenced work.
'''

if __name__ == '__main__':

    nh = 7  # length of the spin chain
    env = MyEnv(nh)   # creates the enviroment

    # designation of state and action spaces

    states = env.observation_space.shape
    actions = env.action_space.n

    lr = 0.01   # learning rate

    n_games = 50000   # complete episodes (from t = 0 to tmax)

    agent = Agent(gamma=0.95, epsilon=1, alpha=lr, nh=nh,
                  n_actions=16, mem_size=40000, batch_size=32,
                  epsilon_end=0.01)

    scores = []  # storage of episode score
    fids = []    # storage of episode maximum fidelity
    tmaxs = []   # storage of time when max. fidelity is reached

    eps_history = []  # storage of epsilon value

    f1 = open("test.dat", "w")  # file to store data
    writer = csv.writer(f1)

    dt = 0.15

    for i in range(n_games):

        done = False
        score = 0
        observation = env.reset()
        fid0 = 0.
        t = 0.
        indt = 0
        tfmax = 0.

        if (i % 1000 == 0):
            fname = 'fid_eps_' + str(i) + '.dat'
            f2 = open(fname, "w")
            writer2 = csv.writer(f2)

        while not done:
            indt += 1
            t = indt*dt
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += np.real(reward)
            agent.remember(observation, action,
                           reward, observation_, int(done))
            observation = observation_
            agent.learn()

            fid = np.real(observation[nh-1]*np.conjugate(observation[nh-1]))

            if (fid > fid0):
                fid0 = np.real(fid)
                tfmax = t

            if (i % 1000 == 0):
                c02 = np.real(observation[0]*np.conjugate(observation[0]))
                row2 = [t, c02, fid]
                writer2.writerow(row2)

        if (i % 1000 == 0):
            f2.close()

        eps_history.append(agent.epsilon)
        scores.append(score)
        tmaxs.append(tfmax)
        fids.append(fid0)

        avg_score = np.mean(scores[max(0, i-100):(i+1)])
        avg_fids = np.mean(fids[max(0, i-100):(i+1)])
        avg_tf = np.mean(tmaxs[max(0, i-100):(i+1)])

        print('episode: ', i, 'score: %.2f' % score,
              'average score %.2f' % avg_score, 'fidelidad: %.2f' % fid)

        row = [i, np.real(fid0), np.real(tfmax), np.real(score),
               np.real(avg_fids), np.real(avg_tf), np.real(avg_score),
               np.real(agent.epsilon)
               ]
        writer.writerow(row)

        if i % 100 == 0 and i > 0:
            agent.save_model()

f1.close()
