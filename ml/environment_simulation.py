import gym
from gym import spaces
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
import os

# import utils


# Absolute path to this file
current_file_path = os.path.abspath(__file__)

# Directory containing this file
current_directory = os.path.dirname(current_file_path)

# Set environment variable
os.environ["PATH_DATASETS"] = current_directory
PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")

# Constants
F = 96485  # Faraday's constant, C/mol
R = 8.314  # Gas constant, J/(mol*K)
T = 298.15  # Temperature, K
FRT = F / (R * T)  # F/RT 값


class DMFCENV(gym.Env):

    def __init__(self, save_dir=os.path.join(PATH_DATASETS, 'logs'), ckpt=(0, 0, 0.), continuous=True) -> None:
        self.save_dir = save_dir

        # Setting action space and observation space
        self.continuous = continuous
        if self.continuous:
            # resting v, working v, resting t, working t
            self.action_space = spaces.Box(low=np.array([0.02, 0.2, 0.31, 0.31]), high=np.array([0.4, 0.65, 1, 2.5]),
                                           dtype=np.float32)
        else:
            self.action_space = spaces.MultiDiscrete([10, 10, 10, 10])  # 10 steps for each action
            self.action_space.low = np.array([0.02, 0.2, 0.31, 0.31])
            self.action_space.high = np.array([0.4, 0.65, 1, 2.5])

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,),
                                            dtype=np.float32)  # no meaning just for shape

        # Setting simulated cell
        self.cell = Cell(0.09, -0., 1e-3, 1e-5, 30 / FRT, 30. / FRT, 1, 500, 0.1)

        # Initialize simulation parameters
        self.l_t = []
        self.l_v = []
        self.l_i = []
        self.reward_plot = []

        self.time = ckpt[2]
        self.current_step = ckpt[1]
        self.cleaning_step = -10
        self.episode_count = ckpt[0]
        self.episode_rewards = []
        self.episode_reward = 0
        self.state = None

    def sample(self, ):
        '''
        sample ranom action from action space
        '''
        if self.continuous:
            return self.action_space.sample()
        else:  # For discrete action space
            return self.action_space.sample() / 9 * (
                        self.action_space.high - self.action_space.low) + self.action_space.low

    def _get_obs(self):
        '''
        get observation from cell
        2d array: (resting resistance, working resistance)
        '''
        # Cell resistance as observation state
        # self.cell.measure_resistance()
        # return self.cell.R
        return self.state

    def reset(self, seed=None, options=None):
        '''
        reset environment
        return resetted state
        '''
        super().reset(seed=seed)
        self.cell.reset()

        self.l_t = []
        self.l_v = []
        self.l_i = []
        self.reward_plot = []
        self.time = 0.
        self.current_step = 0
        self.cleaning_step = -10
        self.episode_count += 1
        self.episode_rewards.append(self.episode_reward)
        self.episode_reward = 0

        observation = self._get_obs()

        return observation

    def step(self, action):
        '''
        conduct experiment using action
        return next state, reward, done, info
        action: np.array or list (resting voltage, working voltage, resting time, working time)
        '''

        # if resting voltage is higher than working voltage, set working voltage to resting voltage
        if action[0] > action[1]:
            action[1] = action[0]
        if isinstance(action, list):
            action = np.array(action)

        # set action variables
        rest_v = action[0]
        work_v = action[1]
        rest_t = int(np.round(10 ** action[2], decimals=0))
        work_t = int(np.round(10 ** action[3], decimals=0))

        # set number of iteration
        iter_t = rest_t + work_t
        # if iter_t < 300:
        #     num_iter = 300 // iter_t
        # else:
        #     num_iter = 1
        num_iter = 300 // iter_t + 1

        # set initial values
        produced_energy = 0
        t_size = 0

        # set initial trajectory
        traj_step_v = []
        traj_step_i = []
        traj_step_t = []

        # iterate resting and working
        for _ in range(num_iter):
            # Resting
            t = np.arange(0, rest_t, 0.1) + self.time  # time grid
            v = np.zeros_like(t) + rest_v  # voltage grid
            t_size += len(t)
            self.time = t[-1] + 0.1
            i, sol = self.cell.operate_cell(rest_v, t)

            traj_step_v.append(v)
            traj_step_i.append(i)
            traj_step_t.append(t)
            # from current trajectory, calculate produced_energy
            produced_energy += np.sum(i[i > 0] * (0.9 - rest_v)) * 0.1

            # Working
            t = np.arange(0, work_t, 0.1) + self.time  # time grid
            v = np.zeros_like(t) + work_v  # voltage grid
            t_size += len(t)
            self.time = t[-1] + 0.1
            i, sol = self.cell.operate_cell(work_v, t)

            traj_step_v.append(v)
            traj_step_i.append(i)
            traj_step_t.append(t)
            # from current trajectory, calculate produced_energy
            produced_energy += np.sum(i[i > 0] * (0.9 - work_v)) * 0.1
            # self.state = np.array([work_v/(i[0]+1e-6), work_v/(i[-1]+1e-6)])

        # stack trajectory
        traj_step_v = np.hstack(traj_step_v)
        traj_step_i = np.hstack(traj_step_i)
        traj_step_t = np.hstack(traj_step_t)
        traj_step = np.vstack([traj_step_t, traj_step_v, traj_step_i])

        # save trajectory in history
        self.l_t.append(traj_step_t)
        self.l_v.append(traj_step_v)
        self.l_i.append(traj_step_i)

        reward_plot = np.zeros(t_size) + produced_energy / (t_size * 0.1)
        self.reward_plot.append(reward_plot)

        # Prepare outputs
        self.episode_reward += produced_energy
        reward = produced_energy / (t_size * 0.1)

        self.current_step += 1
        info = {
            'step_id': f'exp{self.episode_count}_{self.current_step}',
            'step_trajectory': traj_step,
            # 'step_actual_time': utils.get_step_actual_time(action)
        }

        # get info from cell
        # observation = self._get_obs()
        observation = np.vstack([traj_step_v, traj_step_i])[:, :2990]
        # terminated = True if self.time > 7200 else False
        terminated = False

        if terminated:
            self.episode_plot()
            self.final_plot()
            self.reset()

        return observation, reward, terminated, info

    def episode_plot(self):
        '''
        plot current and voltage trajectory of the episode
        '''
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 9), dpi=150)

        # Prepare items to plot
        t_grid = np.hstack(self.l_t)
        v_grid = np.hstack(self.l_v)
        i_grid = np.hstack(self.l_i)
        reward_plot = np.hstack(self.reward_plot)

        axes[0].plot(t_grid, i_grid)
        axes[0].set_xlabel('Time/s')
        axes[0].set_ylabel('Current/mA')
        axes[0].grid()
        axes[1].plot(t_grid, v_grid)
        axes[1].set_xlabel('Time/s')
        axes[1].set_ylabel('Control V/V')
        axes[1].set_ylim(0, 1)
        axes[1].grid()
        axes[2].plot(t_grid, reward_plot)
        axes[2].set_xlabel('Time/s')
        axes[2].set_ylabel('Reward')
        axes[2].set_ylim(0, 1)
        axes[2].grid()
        plt.savefig(os.path.join(self.save_dir, f"EIS_reward_episode_{self.episode_count}.png"))
        plt.close()
        return fig

    def final_plot(self):
        '''
        Plot rewards according to episodes
        '''
        plt.plot(self.episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Episode rewards')
        plt.grid()
        plt.savefig(os.path.join(self.save_dir, f"final.png"))
        plt.close()
        # episode rewards를 txt에 저장
        np.savetxt(os.path.join(self.save_dir, f"episode_rewards.txt"), self.episode_rewards, delimiter=",")


class Cell:
    '''
    Simulated cell class
    '''

    def __init__(self, R0, Rb, J0, J1, k0, k1, C0, C1p, C1n) -> None:
        # 시뮬레이션 파라미터
        self.R0 = R0  # Resistance of R0 (ohm)
        self.Rb = Rb  # Resistance bias of R0 (ohm)
        self.J0 = J0  # Current exchange density of 0 (A/m^2)
        self.J1 = J1  # Current exchange density of 1 (A/m^2)
        self.k0 = k0  # reaction kinetic constant 0 (m/s)
        self.k1 = k1  # reaction kinetic constant 0 (m/s)
        self.C0 = C0  # Capacitance of 0 (mol/m^3)
        self.C1p = C1p  # Forward Capacitance of 1 (mol/m^3)
        self.C1n = C1n  # Backward Capacitance of 1 (mol/m^3)

        self.v_app = 0  # Applied voltage (V)
        self.V = np.array([0., 0.])  # V0, V1 (V)

    def kinetics(self, v, t):
        '''
        Differential equation for calculating voltage change
        v: 2d array, [V0, V1]
        t: time
        '''
        output = np.zeros_like(v)
        i_0 = (self.v_app - v[0] - v[1]) / self.R0 - self.J0 * np.exp(-t / 7200) * (
                    np.exp(v[0] * FRT * self.k0) - np.exp(-v[0] * FRT * (1 - self.k0)))
        if i_0 > 0:
            output[0] = i_0 / self.C0
        else:
            if v[0] + i_0 / self.C0 < 0:
                output[0] = -v[0]
            else:
                output[0] = i_0 / self.C0
        i_1 = (self.v_app - v[0] - v[1]) / self.R0 - self.J1 * np.exp(-t / 7200) * (
                    np.exp(v[1] * self.k1) - np.exp(-v[1] * FRT * (1 - self.k1)))
        if i_1 > 0:
            output[1] = i_1 / self.C1p
        else:
            if v[1] + i_1 / self.C1n < 0:
                output[1] = -v[1]
            else:
                output[1] = i_1 / self.C1n
        return output

    def operate_cell(self, v_app, t_grid):
        '''
        v_app: applied potential (V)
        t_grid: time grid (s)
        '''
        self.v_app = v_app
        sol = odeint(self.kinetics, self.V, t_grid).squeeze()
        self.V = sol[-1]
        i = (self.v_app - sol[:, 0] - sol[:, 1]) / self.R0 - self.Rb
        return i, sol

    def rest(self):
        self.operate_cell(0, np.linspace(0, 1, 300))

    def reset(self):
        self.V = np.array([0., 0.])

    def measure_resistance(self):
        '''
        Measure resistance of the cell as environment state
        R: resistance (Ohm)
        '''
        self.R = []
        t_grid = np.linspace(0, 5, 50)
        for v in [0.65, 0.65]:  # min resting voltage, max working voltage
            i, sol = self.operate_cell(v, t_grid)
            self.R.append(v / np.mean(i))
        self.R = np.array(self.R)
        return self.R