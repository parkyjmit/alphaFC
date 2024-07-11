import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import yaml
from environment_simulation import DMFCENV
from agents import TD3Agent, GAAgent, load_actor, load_target, save_database, load_database
from conduct_rl_training import TD3Actor, TD3Critic  # load_actor function needs this

# Absolute path to this file
current_file_path = os.path.abspath(__file__)

# Directory containing this file
current_directory = os.path.dirname(current_file_path)

# Set environment variable
os.environ["PATH_DATASETS"] = current_directory
PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")


def play_episodes(directory, model, algorithm, obs_size, n_actions, action_low, action_high, agent):
    '''
    This function is used to conduct simulation for episode.
    '''
    count = 0
    while count <= 1000:  # maximum 1000 episodes
        # load actor
        actor = load_actor(directory, model, algorithm, obs_size, n_actions, action_low, action_high)
        target = load_target(directory)
        # play step
        action, reward, terminate = agent.play_step(env, actor, target=target)
        if terminate: count += 1

        # save database
        save_database(directory, agent, env, type='sim')  # to directory from agent.buffer


if __name__ == "__main__":
    # load config file
    args = yaml.load(open(os.path.join(PATH_DATASETS, 'config.yaml'), "r"),
                     Loader=yaml.FullLoader)  # C:/Users/EEL/PycharmProjects/MOR_RL/

    # create directory
    exp_name = args['exp_name']
    algorithm = args['algorithm']
    directory = os.path.join(PATH_DATASETS, args['path'], exp_name)
    os.makedirs(directory, exist_ok=True)

    # load database, env, agent
    database, ckpt = load_database(directory, type='sim')
    model = args['model']

    env = DMFCENV(save_dir=directory, ckpt=ckpt)
    agent = GAAgent(database)

    # get obs_size, n_actions, action_low, action_high
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high

    # play episodes
    play_episodes(directory, model, algorithm, obs_size, n_actions, action_low, action_high, agent)