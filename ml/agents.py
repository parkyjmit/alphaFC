from collections import deque
from typing import Tuple, Union
import gym
import numpy as np
import pandas as pd
import torch
from torch import nn
import os
import warnings
from tqdm import tqdm
from torch import optim
import json
from neural_network_training import GAActor, Experience


class Agent:
    """Base Agent class handeling the interaction with the environment."""

    def __init__(self, replay_buffer: deque) -> None:
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.replay_buffer = replay_buffer
        self.min_reward = -np.inf

    def play_step(self, env: gym.Env, net: Union[nn.Module, object], device: str = "cpu", target: float = 3) -> Tuple[float, bool]:
        """Carries out a single interaction step between the agent and the environment.

        Args:
            net: Actor network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done
        """

        # if the step is a cleaning step, do cleaning
        if env.cleaning_step == env.current_step:
            action = [0.003, 0.103, np.log10(90), np.log10(90)]
        else:
            action = self.get_action(env, net, device, target)
            print(action)

        if isinstance(action, torch.Tensor):
            action = action.squeeze().clone().detach().cpu()
            action = [float(action[0].item()), float(action[1].item()), float(action[2].item()),
                      float(action[3].item())]
        elif isinstance(action, np.ndarray):
            action = [float(action[0]), float(action[1]), float(action[2]), float(action[3])]

        # action = [0.65,0.65,0.31,2.50]  # constant
        # action = [0.02,0.65,0.31,1.778]  # switching
        action_copy = action.copy()
        # do step in the environment
        new_state, reward, done, info = env.step(action_copy)

        if self.min_reward < reward:
            if env.state is not None:
                exp = Experience(env.state, action, reward, done, new_state, info)
                self.replay_buffer.append(exp)

        env.state = new_state
        return action, reward, done


class AlphaFCAgent(Agent):
    def __init__(self, replay_buffer: deque) -> None:
        super().__init__(replay_buffer)
        self.start = 1
        self.end = 0.01
        self.frames = 2000  # how much step to reach end epsilon
        self.global_step = 0

    def get_action(self, env: gym.Env, net: nn.Module, device: str, target: float) -> int:
        """Using the given network, decide what action to carry out using an noisy exploration policy.

        Args:
            net: Actor network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action
        """
        '''
        Get the action from the model
        '''
        if env.state is None or len(self.replay_buffer) < 2:
        #     high = torch.tensor(env.action_space.high, dtype=torch.float32).to(device)
        #     action = high
        #     # action = torch.tensor([0.4, 0.65, 0.31,1.5], dtype=torch.float32).to(device)
        # elif len(self.replay_buffer) < 20:
            # low부터 high까지의 random action
            low = torch.tensor(env.action_space.low, dtype=torch.float32).to(device)
            high = torch.tensor(env.action_space.high, dtype=torch.float32).to(device)
            action = torch.rand(4, dtype=torch.float32).to(device)
            action = action * (high - low) + low
            action = high
        else:
            # 데이터 생성
            x = torch.tensor(env.state, dtype=torch.float32).unsqueeze(0).to(device)
            x = x.repeat(256, 1, 1).to(device)
            target = torch.tensor([target], dtype=torch.float32)

            # Random initialization for searching
            input_a = nn.Parameter(torch.randn((256, 4), dtype=torch.float32), requires_grad=True)

            low = torch.tensor(env.action_space.low, dtype=torch.float32).to(device)
            high = torch.tensor(env.action_space.high, dtype=torch.float32).to(device)

            # optimize batch of input_a
            optimizer = optim.AdamW([input_a], lr=0.01)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
            net = net.to(device)
            net.eval()
            for epoch in tqdm(range(50)):
                optimizer.zero_grad()
                output = net(x, input_a.to(device))
                loss = torch.nn.MSELoss()(output, target)
                loss.backward()
                optimizer.step()
                scheduler.step()
                input_a.data = torch.clamp(input_a.data, low, high)

            # 최종 결과 확인
            with torch.no_grad():
                predicted_output = net(x.float(), input_a.float().to(device))
            margin = torch.abs(predicted_output - target)
            action = input_a[margin.argmin()]
            print("Predicted output", predicted_output[margin.argmin()])
            # print("Margin", margin)
            # print("Candidates", predicted_output)
        return action


def load_database(directory, type='sim'):
    '''
    Load the json database from the directory to Experience objects
    '''
    database_dir = os.path.join(directory, f'{type}_database.json')
    database = deque()
    if os.path.exists(database_dir):
        # json ckpt
        with open(os.path.join(directory, 'ckpt.json'), 'r') as f:
            ckpt = json.load(f)  # start from checkpoint.
        try:
            df = pd.read_json(database_dir)
            for _, d in df.iterrows():
                state = np.array(d['state'], dtype=np.float32)
                action = d['action']
                new_state = np.array(d['new_state'], dtype=np.float32)
                reward = d['reward']
                done = d['done']
                order = d['order']
                database.append(Experience(state, action, reward, done, new_state, order))
        except:
            pass
    else:
        ckpt = (0, 0, 0)  # episode, step, time
    return database, ckpt


def load_actor(directory, n_actions):
    '''
    Load the running actor torch model from the directory
    '''
    actor_dir = os.path.join(directory, 'running_actor.pt')
    print("actor dir", actor_dir)
    actor = GAActor(n_actions)
    if os.path.exists(actor_dir):
        try:
            actor.load_state_dict(torch.load(actor_dir))  # if exists, load the actor\
            print('Success to Load ML model')
        except:
            print('Fail to Load ML model')
            # actor = GAActor(n_actions)  # just for stability
    # else:
    #     actor = GAActor(n_actions)
    actor.eval()
    return actor

def load_target(directory):
    target_dir = os.path.join(directory, 'target.json')
    if os.path.exists(target_dir):
        try:
            target = pd.read_json(target_dir)['Target'][0]
            print('Success to Load Target value')
        except:
            target = 3.0
            print('Fail to Load Target. 3.0 for Target.')
    else:
        target = 3.0
        print('Fail to Load Target. 3.0 for Target.')
    return target

def save_database(directory, agent, env, type='sim'):
    '''
    save replay buffer to json file
    '''
    with warnings.catch_warnings():  # ignore the warning of pandas
        warnings.simplefilter("ignore")
        df = pd.DataFrame(columns=['state', 'action', 'new_state', 'reward', 'done', 'order'])
        for d in agent.replay_buffer:
            e = {}
            e['state'] = d.state
            e['action'] = d.action
            e['new_state'] = d.new_state
            e['reward'] = d.reward
            e['done'] = d.done
            e['order'] = str(d.order)
            df.loc[len(df)] = e
            # df = df.append(e, ignore_index=True)
        df.to_json(os.path.join(directory, f'{type}_database.json'))  # sim_database or exp_database
    with open(os.path.join(directory, 'ckpt.json'), 'w') as f:
        print((env.episode_count, env.current_step, env.time))
        json.dump((env.episode_count, env.current_step, env.time), f)