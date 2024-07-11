import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import gym
from gym import spaces
import numpy as np
from software_control.action import act
from ml import utils


# Absolute path to this file
current_file_path = os.path.abspath(__file__)

# Directory containing this file
current_directory = os.path.dirname(current_file_path)

# Set environment variable
os.environ["PATH_DATASETS"] = current_directory
PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")


class DMFCENV(gym.Env):
    def __init__(self, save_dir=os.path.join('logs'), ckpt=(0, 0, 0)) -> None:
        self.save_dir = save_dir

        self.action_space = spaces.Box(low=np.array([0.02, 0.2, 0.31, 0.31]), high=np.array([0.4, 0.65, 1, 2.5]),
                                       dtype=np.float32)  # resting v, working v, resting t, working t
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)  # no meaning just for shape

        self.time = ckpt[2]
        self.current_step = ckpt[1]
        self.cleaning_step = -10
        self.episode_count = ckpt[0]
        self.episode_rewards = []
        self.episode_reward = 0
        self.state = None

    def _get_obs(self):
        return self.state.astype(np.float32).squeeze()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time = 0.
        self.current_step = 0
        self.cleaning_step = -10
        self.episode_count += 1
        self.episode_rewards.append(self.episode_reward)
        self.episode_reward = 0
        observation = None# self._get_obs()

        return observation

    def step(self, action):
        """
        Takes a step in the environment given an action.

        Args:
            action: A list of four floats representing the agent's action.

        Returns:
            A tuple containing:
                observation (list): The current observation of the environment.
                reward (float): The reward from the previous action.
                terminated (bool): Whether or not the episode terminated after this step.
                info (dict): A dictionary containing information about the step.

        Raises:
            None
        """
        # if self.state is None:
        #     action = [0., 0., 1., 1.]

        # if resting voltage is higher than working voltage, set working voltage to resting voltage
        if action[0] > action[1]:
            action[1] = action[0]
        if isinstance(action, list):
            action = np.array(action)

        self.current_step += 1
        info = {
            'step_id': f'exp{self.episode_count}_{self.current_step}',
            'step_actual_time': utils.get_step_actual_time(action)
        }

        # from action, do experiment
        reward, observation = act(info['step_id'], action, self.save_dir)
        # self.state = self.state.squeeze()
        self.time += info['step_actual_time'][0]
        self.episode_reward += reward

        # observation = self._get_obs()
        terminated = True if self.time > 7200 else False

        if terminated:
            self.reset()

        return observation, reward, terminated, info  # present capacity, produced energy
