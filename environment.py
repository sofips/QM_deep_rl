"""
Definition of actions, environment class and tests for physical properties
"""

import configparser
import numpy as np
from scipy.linalg import expm
import scipy.linalg as la
from gym import Env
from gym.spaces import Discrete, Box

# constant i
comp_i = complex(0, 1)


class MyEnv(Env):
    """
    Class including the environment for the spin chain, i.e., actions,
    dynamics / evolution, and rewards.

    Args:
        config_file (str): Path to the configuration file.

    Attributes:
        n (int): Chain length.
        dt (float): Time step length.
        tolerance (float): Tolerance value.
        max_t_steps (int): Maximum time steps.
        coupling (float): Coupling value.
        field_strength (float): Field strength value.
        n_actions (int): Number of possible actions.
        action_space (gym.spaces.Discrete): Action space.
        observation_space (gym.spaces.Box): Observation space.
        action_mat (numpy.ndarray): Matrices to store actions.
        energies (numpy.ndarray): Energies array.
        bases (numpy.ndarray): Bases array.
        propagators (numpy.ndarray): Propagators array.
        sp_desc (numpy.ndarray): Spectral decomposition array.
        t_step (int): Current time step.
        cstate (numpy.ndarray): Complex state array.
        state (numpy.ndarray): State array.

    """

    def __init__(self, config_file):

        # read system parameters from file
        self.config = configparser.ConfigParser()
        config = self.config

        config.read(config_file)

        self.n = config.getint("system_parameters", "chain_length")
        self.dt = config.getfloat("system_parameters", "tstep_length")
        self.tolerance = config.getfloat("system_parameters", "tolerance")
        self.max_t_steps = config.getint("system_parameters", "max_t_steps")
        self.coupling = config.getfloat("system_parameters", "coupling")
        self.field_strength = config.getfloat("system_parameters", "field_strength")
        self.n_actions = config.getint("system_parameters", "n_actions")
        self.action_set = config.get("system_parameters", "action_set")
        reward_function = config.get("learning_parameters", "reward_function")

        # --------------------------------------------------------------------------
        # Define action and observation space (complex observations)
        # --------------------------------------------------------------------------
        self.action_space = Discrete(self.n_actions)  # 16 acciones posibles
        self.observation_space = Box(low=np.zeros(2 * self.n), high=np.ones(2 * self.n))
        # --------------------------------------------------------------------------
        # Define matrices to store actions, and propagations
        # (also check spect. decomp.)
        # --------------------------------------------------------------------------
        if self.action_set == "zhang":
            self.action_mat = actions_zhang(self.field_strength, self.coupling, self.n)
        elif self.action_set == "oaps":
            self.action_mat = one_field_actions(self.field_strength, self.coupling, self.n)
        else:
            raise ValueError("Invalid action set. Choose 'zhang' or 'oaps'.")

        self.energies = np.zeros((self.n_actions, self.n), dtype=np.complex_)
        self.bases = np.zeros((self.n_actions, self.n, self.n), dtype=np.complex_)
        self.propagators = np.zeros((self.n_actions, self.n, self.n), dtype=np.complex_)
        self.sp_desc = np.zeros((self.n_actions, self.n, self.n), dtype=np.complex_)

        # ---------------------------------------------------------------------------
        # Calculate propagators using exponentials
        # ---------------------------------------------------------------------------

        for i in range(0, self.n_actions):
            self.energies[i, :], self.bases[i, :, :] = la.eig(self.action_mat[i, :, :])
            self.propagators[i, :, :] = expm(-1j * self.action_mat[i] * self.dt)

            for k in range(0, self.n):
                p = np.outer(self.bases[i, :, k], self.bases[i, :, k])
                self.sp_desc[i, :, :] = self.sp_desc[i, :, :] + p * self.energies[i, k]

        # Spectral Decomposition Check
        check_sd(self.n, self.n_actions, self.action_mat, self.sp_desc)
        # Correct Propagation Check
        propagators_check(
            self.n, self.n_actions, self.dt, self.propagators, self.bases, self.energies
        )

        # ---------------------------------------------------------------------------
        # Define reward function
        # ---------------------------------------------------------------------------

        if reward_function == "full reward":
            self.reward = full_reward
        elif reward_function == "original":
            self.reward = original_reward
        elif reward_function == "ipr":
            self.reward = ipr_reward
        else:
            raise ValueError("Invalid reward function. Choose 'full reward', 'original' or 'ipr'.")

        # Set time to 0 and states to one excitation

        self.t_step = 0
        self.cstate = np.zeros(self.n, dtype=np.complex_)
        self.cstate[0] = 1
        self.state = np.zeros(2 * self.n, dtype=np.float_)
        self.state[0] = 1

    def step(self, action):
        """
        Performs a step in the environment.

        Args:
            action (int): The action to take.

        Returns:
            state (numpy.ndarray): The new state with real values.
            cstate (numpy.ndarray): The new complex state.
            t_step (int): The current time step.
            fid (float): The fidelity value.
            reward (float): The reward value.
            done (bool): Whether the episode is done or not.

        """
        config = self.config
        self.t_step = self.t_step + 1

        j = 0
        for i in np.arange(0, self.n):
            self.cstate[i] = complex(self.state[j], self.state[j + 1])
            j += 2

        self.cstate = np.matmul(self.propagators[action, :, :], self.cstate)

        for i in np.arange(0, 2 * self.n, 2):
            self.state[i] = np.real(self.cstate[i // 2])
            self.state[i + 1] = np.imag(self.cstate[i // 2])

        fid = np.real(self.cstate[self.n - 1] * np.conjugate(self.cstate[self.n - 1]))
        
        reward = self.reward(self.state,
                             self.propagators[action, :, :],
                             self.t_step,
                             config)

        if (fid >= 1.0 - self.tolerance) or (self.t_step >= self.max_t_steps):
            done = True
        else:
            done = False
        
        # check state normalization
        if abs(la.norm(self.cstate) - 1.0) > 1e-8:
            raise Exception(f"""State not normalized. 
                             Norm: {la.norm(self.cstate)}""")

        return self.state, self.cstate, self.t_step, fid, reward, done

    def reset(self):
        """
        Resets the environment to the initial state, given by /100000>.
        Sets the time step to 0.

        Returns:
            state (numpy.ndarray): The initial state.
            cstate (numpy.ndarray): The initial complex state.
            t_step (int): The initial time step.

        """

        self.cstate = np.zeros(self.n, dtype=np.complex_)
        self.cstate[0] = 1
        self.state = np.zeros(2 * self.n, dtype=np.float_)
        self.state[0] = 1
        self.t_step = 0

        return self.state, self.cstate, self.t_step


# ----------------------------------------------------------------------#
#                           TESTING FUNCTIONS                          #
# ----------------------------------------------------------------------#


def check_sd(n, n_actions, actions, sd):
    """
    Check the spectral decomposition of actions.

    Parameters:
    - n (int): The size of the matrix.
    - n_actions (int): The number of actions.
    - actions (ndarray): The actions matrices.
    - sd (ndarray): The spectral decomposition matrices.

    Returns:
    - bool: True if the spectral decomposition is correct, False otherwise.
    """

    check_sd = True

    for k in np.arange(0, n_actions):
        for i in np.arange(0, n):
            for j in np.arange(0, n):
                if actions[k, i, j] - sd[k, i, j] > 1e-8:
                    print("error in spectral decomposition")
                    check_sd = False
                    raise Exception("Error in spectral decomposition")

    if check_sd:
        print("Correct Spectral Decomposition")

    return check_sd


def propagators_check(n, n_actions, dt, propagators, bases, energies):
    """
    Check the correctness of state propagation using propagators on the
    eigenvectors of each matrix.

    Args:
        n (int): Number of bases.
        n_actions (int): Number of actions.
        dt (float): Time step.
        propagators (ndarray): Array of shape (n_actions, n, n) representing
        the propagators associated to each action
        bases (ndarray): Array of shape (n_actions, n, n)
        representing the eigenvectors.
        energies (ndarray): Array of shape (n_actions, n) '
        representing the energies / eigenvalues.

    Returns:
        bool: True if state propagation is correct, False otherwise.
    """
    check_prop = True
    comp_i = complex(0, 1)

    for a in np.arange(n_actions):
        for j in np.arange(0, n):
            errores = (
                np.matmul(propagators[a, :, :], bases[a, :, j])
                - np.exp(-comp_i * dt * energies[a, j]) * bases[a, :, j]
            )
            et = np.sum(errores)
            if la.norm(et) > 1e-8:
                check_prop = False
                raise Exception("Error in state propagation")

    if check_prop:
        print("State Propagation: checked")

    return check_prop


# ----------------------------------------------------------------------#
#                      ACTION MATRICES DEFINITION                      #
# ----------------------------------------------------------------------#


def actions_zhang(field_strength, coupling, chain_length):
    """
    Generate matrix of the 16 possible actions
    based on the given parameters. For full definition, check
    the referenced work:
    (https://doi.org/10.1103/PhysRevA.97.052333)
    Args:
        field_strength (float): Magnetic Field value
        coupling (float): Coupling Constants
        chain_length (int): Length of the chain

    Returns:
        numpy.ndarray: A matrix of actions with shape (16, chain_length,
        chain_length).

    """
    nc = 3  # number of control sites, nc=3,there are totally 16 actions.

    # defining action, each row of 'mag' corresponds to one configuration

    def binact(A):  # action label
        m = np.zeros(nc)
        for ii in range(
            nc
        ):  # transfer action to a binary list, for example: action=5, x=[1, 0,
            # 1, 0], the first and third magnetic is on
            m[nc - 1 - ii] = A >= 2 ** (nc - 1 - ii)
            A = A - 2 ** (nc - 1 - ii) * m[nc - 1 - ii]
        return m

    mag = []
    for ii in range(8):  # control at the beginning
        mag.append(
            list(
                np.concatenate(
                    (binact(ii) * field_strength, np.zeros(chain_length - nc))
                )
            )
        )

    for ii in range(1, 8):  # control at the end
        mag.append(
            list(
                np.concatenate(
                    (np.zeros(chain_length - nc), binact(ii) * field_strength)
                )
            )
        )

    mag.append([field_strength for ii in range(chain_length)])

    action_hamiltonians = np.zeros((16, chain_length, chain_length))

    for actions in mag:
        ham = (
            np.diag([coupling for i in range(chain_length - 1)], 1)*(1 - 0j)
            + np.diag([coupling for i in range(chain_length - 1)], -1)*(1 + 0j)
            + np.diag(actions)
        )
        action_hamiltonians[mag.index(actions)] = ham

    return action_hamiltonians

def one_field_actions(bmax, coupling, chain_length):
    """
    Generates a set of action matrices corresponding to fields acting on every individual
    site.

    Parameters:
    bmax (float): The maximum value of the field, used in diagonal elements.
    chain_length (int): Chain length, which defines the size of action matrices.

    Returns:
    numpy.ndarray: A 3D numpy array of shape (chain_length + 1, chain_length, chain_length) representing the action matrices.
    """

    action_matrices = np.zeros((chain_length + 1, chain_length, chain_length))
    J = coupling

    for i in range(0, chain_length):

        for k in range(0, chain_length - 1):
            action_matrices[i + 1, k, k + 1] = J
            action_matrices[i + 1, k + 1, k] = action_matrices[i + 1, k, k + 1]

        action_matrices[i + 1, i, i] = bmax

    for k in range(0, chain_length - 1):
        action_matrices[0, k, k + 1] = J
        action_matrices[0, k + 1, k] = action_matrices[0, k, k + 1]

    return action_matrices


# # ---------------------------------------------------------
#                       Reward functions
# # ---------------------------------------------------------


def full_reward(next_state, prop, time_step, config):

    max_time = config.getint("system_parameters", "max_t_steps")
    gamma = config.getfloat("learning_parameters", "gamma")
    tolerance = config.getfloat("system_parameters", "tolerance")

    fidelity_evolution = state_fidelity(next_state)

    for step in range(time_step, max_time):
        next_state = prop * next_state  # do operation
        fidelity_evolution = np.append(fidelity_evolution,
                                       state_fidelity(next_state))

    max_fid = np.max(fidelity_evolution)

    # reward function
    if max_fid < 0.8:
        reward = max_fid * 10

    if max_fid >= 0.8 and max_fid <= tolerance:
        reward = 100 / (1 + np.exp(10 * (tolerance - max_fid)))

    if max_fid > tolerance:
        reward = 2500 * max_fid

    reward = reward * (gamma**time_step)
    # a discount is given with respected to step

    return reward


def original_reward(next_state, prop, time_step, config):

    gamma = config.getfloat("learning_parameters", "gamma")
    tolerance = config.getfloat("system_parameters", "tolerance")

    fidelity = state_fidelity(next_state)

    if fidelity < 0.8:
        reward = fidelity * 10
    if fidelity >= 0.8 and fidelity <= tolerance:
        reward = 100 / (1 + np.exp(10 * (tolerance - fidelity)))
    if fidelity > tolerance:
        reward = 2500

    reward = reward * (gamma**time_step)

    return reward


def ipr_reward(next_state, prop, time_step, config):

    gamma = config.getfloat("learning_parameters", "gamma")

    ipr = calc_ipr(next_state)
    fidelity = state_fidelity(next_state)
    reward = fidelity / ipr

    reward = reward * (gamma**time_step)

    return reward


def state_fidelity(state):
    return np.asarray((abs(state[-1]) ** 2))  # calculate fidelity


def calc_ipr(state):
    nh = np.shape(state)[0]
    ipr = 0
    for i in range(nh):
        ipr += np.real(state[i] * np.conjugate(state[i])) ** 2
    return 1 / ipr

