import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class LinearQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__()
        self.input_size = input_size
        self.linear1 = nn.Linear(self.input_size*7, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        start_x = x.long().reshape((-1, self.input_size))
        mask = start_x[:, :-1] == 0
        reshaped = F.one_hot(start_x, num_classes=7).reshape((-1, self.input_size*7))
        # print(f"{len(x)=}")
        # print(flattened)
        x = F.relu(self.linear1(reshaped.float()))
        x = self.linear2(x)

        # print(mask.shape, x.shape)
        x = (x + torch.min(x).abs()) * mask + 0.5 * mask
        x /= torch.max(x+0.0001)
        x += 0.0001
        # print(f"{start_x},\n {x}")
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = Path('.') / 'model'
        model_folder_path.mkdir(exist_ok=True)
        file_name = model_folder_path / file_name

        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        path = Path('model', file_name)
        if not path.exists():
            print(f'Given file {path} does not exist!')
            return
        torch.load(path)

class QTrainer:
    def __init__(self, model, lr, gamma) -> None:
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.long)
        next_state = torch.tensor(np.array(next_state), dtype=torch.long)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.long)

        # single_mode = False
        if len(state.shape) == 1:
            # reshape
            # single_mode = True
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        # print(len(done), target, state)
        for idx in range(len(done)):
            q_new = reward[idx]
            if not done[idx]:
                q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # print(target[idx])
            # print(len(done), target.shape)
            arg_max = torch.argmax(action[idx]).item()
            target[idx][arg_max] = q_new

        # r + y * max(next pred Q val)
        # q_new = pred.clone()
        # preds[argmax]
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()

