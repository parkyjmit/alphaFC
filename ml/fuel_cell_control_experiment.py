import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from knockknock import slack_sender
import yaml
from environment_realworld import DMFCENV
from agents import AlphaFCAgent, load_actor, load_target, save_database, load_database

# Absolute path to this file
current_file_path = os.path.abspath(__file__)

# Directory containing this file
current_directory = os.path.dirname(current_file_path)

# Set environment variable
os.environ["PATH_DATASETS"] = current_directory
PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")

webhook_url = 'your_webhook_url'
@slack_sender(webhook_url=webhook_url, channel='# YOUR_CHANNEL', user_mentions=['YOUR_NAME'])
def play_episode(directory, n_actions, agent, running_time):
    '''
    This function is called by the slack_sender decorator.
    Send email when episode is finished.
    '''
    import time
    start = time.time() 
    end = 0
    while end < running_time:
        # load actor and target
        actor = load_actor(directory, n_actions)
        target = load_target(directory)
        # play step
        action, reward, terminate = agent.play_step(env, actor, target=target)
        print('Reward: ', reward)
        # save database
        save_database(directory, agent, env, type='real')
        # if terminate: break
        end = time.time() - start
    return "Episode end!"


if __name__ == "__main__":
    # load config.yaml
    args = yaml.load(open(os.path.join(PATH_DATASETS, 'config.yaml'), "r"),
                     Loader=yaml.FullLoader) 

    # create directory
    exp_name = args['exp_name']
    algorithm = args['algorithm']
    directory = os.path.join(args['path'], exp_name)
    os.makedirs(directory, exist_ok=True)

    # load database, env, agent
    database, ckpt = load_database(directory, type='real')
    model = args['model']

    env = DMFCENV(save_dir=directory, ckpt=ckpt)
    agent = AlphaFCAgent(database)

    # get obs_size, n_actions, action_low, action_high
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high

    # play episode
    play_episode(directory, n_actions, agent, args['running_time'])
    