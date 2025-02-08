#####################################################################
# Generate configuration files for RL experiments with ODC
#####################################################################

import configparser
import sys

cname = sys.argv[1]
config = configparser.ConfigParser()

experiment_description = (
    """ Experiment to test the performance of a reinforcement learning 
        agent using pytorch.
    """
)

new_experiment = input("New experiment? (y/n): ")
new_experiment = True if new_experiment == "y" else False if new_experiment == "n" else sys.exit("Error: y o n")
experiment_name = input("Experiment name/name: ")
run = input("Run experiment (y/n): ")
run = True if run == "y" else False if run == "n" else sys.exit("Error: y o n")


# -----------------------------------------------------------------#
#                         SYSTEM PARAMETERS                       #
# -----------------------------------------------------------------#

chain_length = 10
tstep_length = 0.15
tolerance = 0.05
max_t_steps = 50
field_strength = 100
coupling = 1
n_actions = 16

# -----------------------------------------------------------------#
#                    LEARNING HYPERPARAMETERS                     #
# -----------------------------------------------------------------#

number_of_features = 2 * chain_length
number_of_episodes = 50000
step_learning_interval = 5

learning_rate = 0.001
gamma = 0.95

# memory
replace_target_iter = 200
memory_size = 40000
batch_size = 32

# epsilon
epsilon = 0.99
epsilon_decay = 0.0001
epsilon_minimum = 0.01

# dqn
fc1_dims = 50
fc2_dims = 50
dropout = 0.0


reward_function = "original"  # "original" , "full reward", "ipr"
action_set = "zhang"  # "zhang", "oaps" (action per site)
n_actions = (
    16 if action_set == "zhang" else chain_length + 1
    if action_set == "oaps" 
    else 0
)

# ---------------------------------------------------
config["experiment"] = {"new_experiment": str(new_experiment),
                        "experiment_name": experiment_name,}


config["system_parameters"] = {
    "chain_length": str(chain_length),
    "tstep_length": str(tstep_length),
    "tolerance": str(tolerance),
    "max_t_steps": str(max_t_steps),
    "field_strength": str(field_strength),
    "coupling": str(coupling),
    "n_actions": str(n_actions),
    "action_set": action_set,
}

config["learning_parameters"] = {
    "number_of_features": str(number_of_features),
    "number_of_episodes": str(number_of_episodes),
    "learning_rate": str(learning_rate),
    "gamma": str(gamma),
    "replace_target_iter": str(replace_target_iter),
    "memory_size": str(memory_size),
    "batch_size": str(batch_size),
    "epsilon": str(epsilon),
    "epsilon_decay": str(epsilon_decay),
    "epsilon_minimum": str(epsilon_minimum),
    "fc1_dims": str(fc1_dims),
    "fc2_dims": str(fc2_dims),
    "dropout": str(dropout),
    "reward_function": reward_function,
}

config["tags"] = {
    "reward_function": reward_function,
    "mlflow.note.content": experiment_description,
    "action set": action_set,
    "chain_length": str(chain_length),
}
# ---------------------------------------------------

config_name = cname + ".ini"

with open(config_name, "w") as configfile:
    config.write(configfile)
