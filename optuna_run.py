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
import optuna

optuna.logging.set_verbosity(optuna.logging.ERROR)
from optuna.integration.mlflow import MLflowCallback

"""
Optimization of hyperparameters using Optuna.
"""
# set mlflow uri and experiment
tracking_uri = "http://127.0.0.1:5768"
mlflow.set_tracking_uri(tracking_uri)

# access configuration file
config_file = sys.argv[1]
config = configparser.ConfigParser()
config.read(config_file)

# experiment parameters for mlflow configuration
now = datetime.datetime.now()
date_str = now.strftime("%Y%m%d_%H%M%S")
experiment_name = f"opt_qmdrl_{date_str}"



def champion_callback(study, frozen_trial):
    """
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values.

    Note: This callback is not intended for use in distributed computing systems such as Spark
    or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
    workers or agents.
    The race conditions with file system state management for distributed trials will render
    inconsistent values with this callback.
    """

    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (
                abs(winner - study.best_value) / study.best_value
            ) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(
                f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}"
            )


client = MlflowClient(tracking_uri=tracking_uri)
experiment = client.create_experiment(name=experiment_name)
experiment = mlflow.get_experiment_by_name(experiment_name)
mlflow.set_experiment(experiment_name)

def run_state(config, agent, env):

    # initialize variables to save results
    scores = []
    fid_max_vector = []
    t_fid_max_vector = []
    fid_end_vector = []
    t_end_vector = []
    eps_history = []
    cpu_time_history = []

    # writer = csv.writer(f1, delimiter=" ")
    # action_writer = csv.writer(f2, delimiter=" ")

    stp = 0
    number_of_episodes = config.getint("learning_parameters", "number_of_episodes")

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
            obs_state_, obs_cstate_, t_step, fidelity, reward, done = env.step(action)
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

        avg_score = np.mean(scores[max(0, i - bin_size) : (i + 1)])
        avg_fid_max = np.mean(fid_max_vector[max(0, i - bin_size) : (i + 1)])
        avg_time_fid_max = np.mean(t_fid_max_vector[max(0, i - bin_size) : (i + 1)])
        avg_fid_end = np.mean(fid_end_vector[max(0, i - bin_size) : (i + 1)])
        avg_time_end = np.mean(t_end_vector[max(0, i - bin_size) : (i + 1)])
        avg_cpu_time = np.mean(cpu_time_history[max(0, i - bin_size) : (i + 1)])

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
                step=int(i // bin_size),
            )
            mlflow.log_metric(
                "final_fidelity",
                avg_fid_end,
                step=int(i // bin_size),
            )
            mlflow.log_metric(
                "t_fid_max_vector",
                avg_time_fid_max,
                step=int(i // bin_size),
            )
            mlflow.log_metric(
                "t_end_vector",
                avg_time_end,
                step=int(i // bin_size),
            )
            mlflow.log_metric(
                "Qvalue",
                avg_score,
                step=int(i // bin_size),
            )
            mlflow.log_metric(
                "cpu_time",
                avg_cpu_time,
                step=int(i // bin_size),
            )
        # row = [
        #     i,
        #     np.real(fid_max),
        #     np.real(t_fid_max),
        #     np.real(fidelity),
        #     np.real(t_step),
        #     np.real(score),
        #     np.real(agent.epsilon),
        # ]
        #writer.writerow(row)

        # if fid_max > config.getfloat("system_parameters", "tolerance"):
        #     action_sequence.append(fidelity)
        #     action_writer.writerow(action_sequence)

        mlflow.pytorch.log_model(agent.Q_eval, "model")

   

    return (
        eps_history,
        scores,
        fid_max_vector,
        t_fid_max_vector,
        fid_end_vector,
        t_end_vector,
        cpu_time_history,
    )


def objective(trial):
    step = 0
    run_name = f'run_step{step}'

    with mlflow.start_run(nested=True,run_name=run_name):
        step += 1
        run_name = f'run_step{step}'

        soft_success_training_rate = 0
        true_success_training_rate = 0
        config_instance = config
        # Define hyperparameters using trial.suggest_* methods
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        fc1_dims = trial.suggest_int("fc1_dims", 32, 512, log=True)
        config_instance.set("learning_parameters", "learning_rate", str(learning_rate))
        config_instance.set("learning_parameters", "fc1_dims", str(fc1_dims))

        # initialize environment and agent
        env = MyEnv(config_instance)
        agent = Agent(config_instance)

        # Convert the configuration to a dictionary
        parameters = {
            section: dict(config_instance.items(section))
            for section in config_instance.sections()
        }
        
        experiment_tags = parameters["tags"]
        system_parameters = parameters["system_parameters"]
        learning_parameters = parameters["learning_parameters"]
                
        mlflow.log_params(learning_parameters)
        mlflow.log_params(system_parameters)
        mlflow.set_tags(experiment_tags)
        
        # run the state
        (
            eps_history,
            scores,
            fid_max_vector,
            t_fid_max_vector,
            fid_end_vector,
            t_end_vector,
            cpu_time_history,
        ) = run_state(config_instance, agent, env)

        for fid in fid_max_vector:
            if fid > 0.9:
                soft_success_training_rate += 1
            if fid >= 0.95:  
                true_success_training_rate += 1
        
        soft_success_training_rate = soft_success_training_rate / len(fid_max_vector)
        true_success_training_rate = true_success_training_rate / len(fid_max_vector)
        max_fid = np.max(fid_max_vector)
        avg_Qvalue = np.mean(scores)

        mlflow.log_metric("soft_success_training_rate", soft_success_training_rate)
        mlflow.log_metric("true_success_training_rate", true_success_training_rate)
        mlflow.log_metric("max_fid", max_fid)
        mlflow.log_metric("avg_Qvalue", avg_Qvalue)

        # Report to Optuna (logs intermediate results)
        trial.report(max_fid, step)

        # Stop early if not promising
        if trial.should_prune():
            mlflow.log_metric("pruned_at_step", step)
            raise optuna.exceptions.TrialPruned()

        return true_success_training_rate
    


run_name = "third_attempt"

# Initiate the parent run and call the hyperparameter tuning child run logic
with mlflow.start_run(
    experiment_id=experiment.experiment_id, run_name=run_name
):
    # Initialize the Optuna study
    study = optuna.create_study(direction="maximize")

    # Execute the hyperparameter optimization trials.
    # Note the addition of the `champion_callback` inclusion to control our logging
    study.optimize(objective, n_trials=50)

    parameters = {section: dict(config.items(section)) for section in config.sections()}

    mlflow.log_params(parameters)
    mlflow.log_params(study.best_params)

    # Retrieve the best trial
    best_trial = study.best_trial

    # Log additional metrics from the best trial
    mlflow.log_metric(
        "best_true_success_training_rate", best_trial.value
    )  # Main metric
    mlflow.log_metric(
        "best_soft_success_training_rate",
        best_trial.user_attrs.get("soft_success_training_rate", 0),
    )
    mlflow.log_metric("best_max_fid", best_trial.user_attrs.get("max_fid", 0))
    mlflow.log_metric("best_avg_Qvalue", best_trial.user_attrs.get("avg_Qvalue", 0))