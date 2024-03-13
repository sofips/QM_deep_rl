from state_env import State
from state_env_sp import *
import sys
from RL_brain_easy import DeepQNetwork
# updated neural network, for using this version,
from RL_brain_pi_deep import DQNPrioritizedReplay
#  change the definition of "RL" correspondingly

import numpy as np
from scipy import *
import matplotlib.pyplot as plt
import csv


def run_state():
    stp = 0

    lth = 28  # test for "STAT_NUM=6" in state_env
    # lth=50 # test for "STAT_NUM=11" in state_env, the larger STAT_NUM, the larger 'lth'

    successtime = []  # time to reach fidelity 0.95 (or not reached)
    actionspace = []
    fids = []  # fidelity
    Qvalue = []  # total reward
    tmaxs = []

    f1 = open('PRE_sp_old.dat', "w")
    writer = csv.writer(f1, delimiter= ' ')
    indt = 0

    for episode in range(50000):
        observation = env.reset()
        observation_ = env.reset()
        newaction = []
        Q = 0
        fid0 = 0
        for i in range(lth):  # episode maximum length

            # observation: input of network; action, the chosen action
            action = RL.choose_action(observation)
            newaction.append(action)
            # observation_:new action; done:break the episode or not
            #print('obs',observation,'obs_',observation_)
            observation =  observation_.copy()
            observation_, reward, done, fidelity = env.step(action)
            #print('obs post step',observation,'obs_ post step',observation_)

            #print(observation_, reward, done, fidelity)
            # store this cluster in memory
            RL.store_transition(observation, action, reward, observation_)
            Q += reward  # total reward
            if (stp > 500) and (stp % 5 == 0):
                RL.learn()  # update neural network

            # update current state(input of the network)
            #observation_old = observation_.copy()
            if (fidelity > fid0):
                fid0 = np.real(fidelity)
                tfmax = i*0.15

            if done:  # fidelity(reward) larger than threshold
                newaction += [0 for xx in range(lth-len(newaction))]
                print(str(i+1)+'  '+str(episode) + '  ' + str(fidelity))
                successtime.append(i+1)
                # fids.append(fidelity)
                actionspace.append(newaction)
                Qvalue.append(Q)

                break
            stp += 1
            i += 1
            if i == lth-1:
                successtime.append(i+1)
                actionspace.append(newaction)
                # fids.append(fidelity)
                Qvalue.append(Q)
                print('fail'+'  '+str(episode)+'  '+str(fidelity))
                #print(actionspace)
        tmaxs.append(tfmax)
        fids.append(fid0)

        avg_fids = np.mean(fids[max(0, episode-100):(episode+1)])
        avg_tf = np.mean(tmaxs[max(0, episode-100):(episode+1)])

        print('episode: ', episode, 'average time %.2f' % avg_tf, 'fidelidad: %.2f' % fid0,
              'fid. media: %.2f' % avg_fids, 'epsilon: %.2f' % RL.epsilon)

        row = [episode, np.real(fid0), np.real(tfmax),
               np.real(avg_fids), np.real(avg_tf)]
        writer.writerow(row)
    return successtime, actionspace, fids, Qvalue


# begin here
if __name__ == "__main__":

    enviroment = sys.argv[1]
    #env = MyEnv()
    if enviroment == 'sp':
        env = MyEnv()
    else:
        env = State()

    RL = DQNPrioritizedReplay(env.n_actions, env.n_features,  # updated network, it change the way of learning
                              learning_rate=0.01,
                              reward_decay=0.95,
                              e_greedy=0.99,
                              replace_target_iter=200,
                              memory_size=40000,
                              e_greedy_increment=0.0001,
                              prioritized=True,
                              rus=None
                              )

    # RL = DeepQNetwork(env.n_actions, env.n_features,
    #                           learning_rate=0.01,
    #                           reward_decay=0.95,  #for nth step of each episode, the reward is r*(0.95^n)
    #                           e_greedy=0.99,      #finally, we choose the network's choice with 0.99 probability
    #                           replace_target_iter=200,  #every 200 times of learning, update the eveluation network
    #                           memory_size=40000,
    #                           e_greedy_increment=0.0001, #every learning step, increase the probability of trusting networ
    #                           )

    r = run_state()

