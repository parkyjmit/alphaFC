from collections import OrderedDict, deque, namedtuple
from typing import Any, Iterator, List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import time

# Absolute path to this file
current_file_path = os.path.abspath(__file__)

# Directory containing this file
current_directory = os.path.dirname(current_file_path)

# Set environment variable
os.environ["PATH_DATASETS"] = current_directory
PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")

# Named tuple for storing experience steps gathered in training
Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "done", "new_state", "order"],
)  # for memory efficiency


class CNN1D(nn.Module):
    def __init__(self, hidden):
        super(CNN1D, self).__init__()
        self.hidden = hidden
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=16, stride=4)  # 2, 2990 -> 32, 744
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=4)  # 32, 744 -> 32, 186
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8)  # 32, 186 -> 64, 179
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=4)  # 64, 179 -> 64, 44
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=hidden, kernel_size=4)  # 64, 44 -> 128, 41
        self.batchnorm3 = nn.BatchNorm1d(hidden)
        self.pool3 = nn.MaxPool1d(kernel_size=4)  # 128, 41 -> 128, 10
        self.fc1 = nn.Linear(hidden * 10, hidden)  # 1280 -> 128
        self.fc2 = nn.Linear(hidden, hidden)

    def forward(self, x):
        x = self.pool1(F.relu(self.batchnorm1(self.conv1(x))))
        x = self.pool2(F.relu(self.batchnorm2(self.conv2(x))))
        x = self.pool3(F.relu(self.batchnorm3(self.conv3(x))))
        x = x.view(-1, self.hidden * 10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class AlphaFCCritic(nn.Module):
    def __init__(self, n_actions) -> None:
        super().__init__()
        hidden = 128
        self.cnn = CNN1D(hidden)
        self.fc1 = nn.Linear(n_actions, hidden)
        self.fc2 = nn.Linear(2*hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)

    def forward(self, x, a):
        x = self.cnn(x)
        a = F.relu(self.fc1(a))
        x = torch.cat((x, a), dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn from them.

    Args:
        capacity: size of the buffer
    """

    def __init__(self, dir) -> None:
        self.buffer = deque(maxlen=int(1e5))
        df = pd.read_json(os.path.join(dir, 'real_database.json'))  # load database

        # Store experiences
        for _, d in df.iterrows():
            state = d['state']
            action = d['action']
            new_state = d['new_state']
            reward = d['reward']
            done = d['done']
            order = d['order']
            self.buffer.append(Experience(state, action, reward, done, new_state, order))

    def __len__(self) -> None:
        return len(self.buffer)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)  # cannot be selected multiple times
        states, actions, rewards, dones, next_states, _ = zip(*(self.buffer[idx] for idx in indices))

        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.float32),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.bool),
            torch.tensor(next_states, dtype=torch.float32),
        )


class MLData(torch.utils.data.Dataset):
    '''
    Dataset for DMFC
    Trajectory data: (2, 2990)
        it should be used as next state
    State data: (2,)
    Action data: (4,)
    Reward data: (1,)
    '''

    def __init__(self, data_dir):
        df = pd.read_json(os.path.join(data_dir, 'real_database.json'))  # load database
        self.df = df[df['state'].apply(lambda x: np.array(x).shape == (2, 2990))]

        idx = [2578]  # incontinuous index
        # split dataframe according to idx
        df_splits = []
        for i in range(len(idx)+1):  # +1 for the last part
            if i == 0:
                df_splits.append(self.df.loc[:idx[i]-1])
            elif i == len(idx):
                df_splits.append(self.df.loc[idx[i-1]:])
            else:
                df_splits.append(self.df.loc[idx[i-1]:idx[i]-1])
        Xs = []
        actions = []
        Ys = []
        for df in df_splits:
            Xs.append(df['state'].to_numpy()[1:])
            actions.append(df['action'].to_numpy()[:-1])
            Ys.append(df['reward'].to_numpy()[:-1])
            Xs.append(df['state'].to_numpy()[:-1])
            actions.append(df['action'].to_numpy()[1:])
            Ys.append(df['reward'].to_numpy()[1:])
        for df in df_splits:
            Xs.append(df['state'].to_numpy())
            actions.append(df['action'].to_numpy())
            Ys.append(df['reward'].to_numpy())
        self.X = np.concatenate(Xs)
        self.action = np.concatenate(actions)
        self.Y = np.concatenate(Ys)
        self.len = len(self.X)

    def __getitem__(self, index):
        return np.array(self.X[index]).astype(np.float32), \
            np.array(self.action[index]).astype(np.float32), \
            self.Y[index].astype(np.float32)

    def __len__(self):
        return self.len


if __name__ == "__main__":
    # Load config file
    args = yaml.load(open(os.path.join(PATH_DATASETS, 'config.yaml'), "r"), Loader=yaml.FullLoader)

    # Create experiment directory
    exp_name = args['exp_name']
    directory = os.path.join(args['path'], exp_name)
    os.makedirs(directory, exist_ok=True)
    args['dir'] = directory

    # Set logger and callback

    while True:
        print('Train loop')
        # Load actor model if exists

        model = AlphaFCCritic(n_actions=4)
        # if os.path.exists(os.path.join(directory, 'model.pt')):
        #     model.load_state_dict(torch.load(os.path.join(directory, 'model.pt')))

        print('Load model. done.')
        mldata = MLData(directory)

        # split train, val, test
        train_size = int(0.9 * len(mldata))
        val_size = len(mldata) - train_size
        # loader
        train_dataset, val_dataset = torch.utils.data.random_split(mldata,
                                                        [train_size, val_size])
        train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=True)

        # Setup
        print('Load data. done.')

        # Define the optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True, factor=0.5)

        # Train the model
        n_epochs = 300
        train_loss_list = []
        val_loss_list = []
        min_val = torch.inf
        # early stopping
        patience = 50
        cnt = 0

        print('Train start')
        for epoch in range(n_epochs):
            # keep track of training and validation loss
            train_loss = 0.0
            val_loss = 0.0

            # Training the model
            model.train()
            for batch in tqdm(train_loader):
                X, action, Y = batch
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(X, action)
                # calculate the batch loss
                loss = torch.nn.MSELoss()(output.squeeze(), Y)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # update training loss
                train_loss += loss.item() * X.size(0)

            # validate the model
            model.eval()
            for batch in val_loader:
                X, action, Y = batch
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(X, action)
                # calculate the batch loss
                loss = nn.L1Loss()(output.squeeze(), Y)
                # update average validation loss
                val_loss += loss.item() * X.size(0)

            # calculate average losses
            train_loss = train_loss / len(train_loader.dataset)
            val_loss = val_loss / len(val_loader.dataset)
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            scheduler.step(val_loss)

            if val_loss < min_val:
                min_val = val_loss
                torch.save(model.state_dict(), os.path.join(directory, 'running_model.pt'))
                cnt = 0
            else:
                cnt += 1
            if cnt == patience:
                print(f'Early stopping at epoch {epoch}')
                break
            # print training/validation statistics
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch + 1, train_loss, val_loss))

        if min_val < 0.011:
            model.load_state_dict(torch.load(os.path.join(directory, 'running_model.pt')))
            torch.save(model.state_dict(), os.path.join(directory, 'running_actor.pt'))
        time.sleep(300*16)
