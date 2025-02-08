import sys
import numpy as np
import csv
import configparser
from environment import MyEnv
from agent import Agent
import os

"""
Implementation of the algorithm developed in the work of Zhang et.
al (https://doi.org/10.1103/PhysRevA.97.052333). The following
code uses Deep Reinforcement Learning to obtain an optimal
sequence of magnetic fields that should be applied to the
extremes of a spin chain in order to achieve a perfect
transmission.

Arguments:
 - config_file: configuration file with the parameters of the system and
 the reinforcement learning agent
 - directory: directory to save the results and the configuration file.
"""

# access configuration file
config_file = sys.argv[1]
config = configparser.ConfigParser()
config.read(config_file)

# create directory to save results
directory = sys.argv[2]
isExist = os.path.exists(directory)

if not isExist:
    os.mkdir(directory)
else:
    print("Warning: Directory already exists")
    exit()

# save config file in created directory
cmd = f'cp "{config_file}" "{directory}"'
os.system(cmd)

# generate files to save results of learning and successfull actions
filename = directory + "/results.dat"
f1 = open(filename, "w")
filename = directory + "/actions.dat"
f2 = open(filename, "a")

# initialize environment
env = MyEnv(config_file)
nh = env.n

# initialize agent
agent = Agent(config_file)
number_of_episodes = config.getint("learning_parameters", "number_of_episodes")

# initialize variables to save results
scores = []
fid_max_vector = []
t_fid_max_vector = []
fid_end_vector = []
t_end_vector = []
eps_history = []

writer = csv.writer(f1, delimiter=" ")
action_writer = csv.writer(f2, delimiter=" ")
stp = 0
for i in range(number_of_episodes):

    done = False
    score = 0
    obs_state, obs_cstate, t_step = env.reset()
    fid_max = 0.0
    final_fid = 0.0

    t = 0.0
    t_fid_max = 0.0
    t_fid_final = 0.0
    action_sequence = []

    while not done:
        action = agent.choose_action(obs_state)
        obs_state_, obs_cstate_, t_step, fidelity, reward, done = env.step(
            action)
        score += np.real(reward)
        agent.store_transition(obs_state, action, reward, obs_state_, done)
        obs_state = obs_state_.copy()
        obs_cstate = obs_cstate_.copy()

        action_sequence.append(action)

        if stp > 500 and stp % 5 == 0:
            agent.learn()

        if fidelity > fid_max:
            fid_max = np.real(fidelity)
            t_fid_max = t_step

        stp += 1

    eps_history.append(agent.epsilon)
    scores.append(score)
    t_fid_max_vector.append(t_fid_max)
    fid_max_vector.append(fid_max)
    fid_end_vector.append(fidelity)
    t_end_vector.append(t_step)

    avg_score = np.mean(scores[max(0, i - 100):(i + 1)])
    avg_fid_max = np.mean(fid_max_vector[max(0, i - 100):(i + 1)])
    avg_time_fid_max = np.mean(t_fid_max_vector[max(0, i - 100):(i + 1)])
    avg_fid_end = np.mean(fid_end_vector[max(0, i - 100):(i + 1)])
    avg_time_end = np.mean(t_end_vector[max(0, i - 100):(i + 1)])

    print(
        "episode: ",
        i,
        "score: %.2f" % score,
        "average score %.2f" % avg_score,
        "fidelidad final: %.2f" % fidelity,
        "fid. media final: %.2f" % avg_fid_end,
        "fidelidad maxima: %.2f" % fidelity,
        "fid. media maxima: %.2f" % avg_fid_max,
        "epsilon: %.2f" % agent.epsilon,
    )

    row = [
        i,
        np.real(fid_max),
        np.real(t_fid_max),
        np.real(fidelity),
        np.real(t_step),
        np.real(score),
        np.real(agent.epsilon),
    ]
    writer.writerow(row)

    if fid_max > config.getfloat('system_parameters','tolerance'):
        action_sequence.append(fidelity)
        action_writer.writerow(action_sequence)


f1.close()
