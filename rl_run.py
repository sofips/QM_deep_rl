import sys
import numpy as np
import csv
import configparser
from environment import MyEnv
from agent import Agent
import os
import datetime
import mlflow
from mlflow.tracking import MlflowClient
import time
import torch as T
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
"""


# access configuration file
config_file = sys.argv[1]
config = configparser.ConfigParser()
config.read(config_file)

# experiment parameters for mlflow configuration
experiment_name = config.get("experiment", "experiment_name")
now = datetime.datetime.now()
date_str = now.strftime("%Y%m%d_%H%M%S")
run_name = f"{experiment_name}_{date_str}"

# create directory to save results
directory = "drl_results/" + run_name

try:
    os.mkdir(directory)
except OSError:
    raise RuntimeError("Creation of the directory %s failed. Maybe it already exists" % directory)

# save config file in created directory
cmd = f'cp "{config_file}" "{directory}"'
os.system(cmd)

# Convert the configuration file a dictionary
parameters = {section: dict(config.items(section)) for section in config.sections()}
experiment_tags = parameters["tags"]
system_parameters = parameters["system_parameters"]
learning_parameters = parameters["learning_parameters"]

# generate files to save results of learning and successfull actions
filename = directory + "/results.dat"
f1 = open(filename, "w")
filename = directory + "/actions.dat"
f2 = open(filename, "w")

tracking_uri = "http://127.0.0.1:5005"
client = MlflowClient(tracking_uri=tracking_uri)

new_experiment = config.getboolean("experiment", "new_experiment")
if new_experiment:
    print(f"Creating new experiment: {new_experiment}")
    experiment = client.create_experiment(name=experiment_name, tags=experiment_tags)
    print(f"Experiment ID: {experiment}")

experiment = mlflow.get_experiment_by_name(experiment_name)
print(f"Experiment ID: {experiment.experiment_id}")
# initialize environment and agent
env = MyEnv(config_file)
agent = Agent(config_file)

# initialize variables to save results
scores = []
fid_max_vector = []
t_fid_max_vector = []
fid_end_vector = []
t_end_vector = []
eps_history = []
cpu_time_history = []

writer = csv.writer(f1, delimiter=" ")
action_writer = csv.writer(f2, delimiter=" ")

stp = 0
number_of_episodes = config.getint("learning_parameters", "number_of_episodes")
mlflow.set_tracking_uri(uri=tracking_uri)
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name=run_name,nested=False):
    # mlflow.log_params(system_parameters)
    # mlflow.log_params(learning_parameters)
    # mlflow.set_tags(experiment_tags)
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

        t1 = time.time()
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
        t2 = time.time()

        eps_history.append(agent.epsilon)
        scores.append(score)
        t_fid_max_vector.append(t_fid_max)
        fid_max_vector.append(fid_max)
        fid_end_vector.append(fidelity)
        t_end_vector.append(t_step)
        cpu_time_history.append(t2 - t1)

        # bin metrics to log average values
        bin_size = 100

        avg_score = np.mean(scores[max(0, i - bin_size):(i + 1)])
        avg_fid_max = np.mean(fid_max_vector[max(0, i - bin_size):(i + 1)])
        avg_time_fid_max = np.mean(t_fid_max_vector[max(0, i - bin_size):(i + 1)])
        avg_fid_end = np.mean(fid_end_vector[max(0, i - bin_size):(i + 1)])
        avg_time_end = np.mean(t_end_vector[max(0, i - bin_size):(i + 1)])
        avg_cpu_time = np.mean(cpu_time_history[max(0, i - bin_size):(i + 1)])
        
        print(
            "episode: ",
            i,
            "score: %.2f" % score,
            "average score %.2f" % avg_score,
            "fidelidad final: %.2f" % fidelity,
            "fid. media final: %.2f" % avg_fid_end,
            "fidelidad maxima: %.2f" % fid_max,
            "fid. media maxima: %.2f" % avg_fid_max,
            "epsilon: %.2f" % agent.epsilon,
        )

        if i % bin_size == 0:
            mlflow.log_metric(
                "max_fidelity",
                avg_fid_max,
                step=int(i//bin_size),
            )
            mlflow.log_metric(
                "final_fidelity",
                avg_fid_end,
                step=int(i//bin_size),
            )
            mlflow.log_metric(
                "t_fid_max_vector",
                avg_time_fid_max,
                step=int(i//bin_size),
            )
            mlflow.log_metric(
                "t_end_vector",
                avg_time_end,            
                step=int(i//bin_size),
            )
            mlflow.log_metric(
                "Qvalue",
                avg_score,
                step=int(i//bin_size),
            )
            mlflow.log_metric(
                "cpu_time",
                avg_cpu_time,
                step=int(i//bin_size),
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

    mlflow.pytorch.log_model(agent.Q_eval, "model")
    num_gpus = T.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    f1.close()
