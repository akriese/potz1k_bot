from collections import deque
from IPython import display
import matplotlib.pyplot as plt

import numpy as np
import random
import torch

from game import PotzAI
from model import QTrainer, LinearQNet

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self, n=3) -> None:
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.n = n
        self.model = LinearQNet(self.n*self.n+1, 256, self.n*self.n)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        return game.get_state()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memeory(self):
        mini_sample = random.sample(self.memory, BATCH_SIZE) if len(self.memory) > BATCH_SIZE else self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memeory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def possible_moves(self, state):
        return np.where(state[:-1] == 0)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = np.zeros(self.n*self.n)
        possible_moves = self.possible_moves(state)[0]
        if random.randint(0, 200) < self.epsilon:
            move = random.choice(possible_moves)
            final_move[move] = 1
            if final_move.sum() != 1:
                print(f"{move=}, {state=}, {possible_moves=}")
        else:
            state0 = torch.tensor(state, dtype=torch.long)
            prediction = self.model(state0)
            # print(f"{prediction.shape=}")
            arg_max = torch.argmax(prediction[0][possible_moves]).item()
            move = possible_moves[arg_max]
            final_move[move] = 1
            if final_move.sum() != 1:
                print(f"{move=}, {state=}, {prediction=}")


        return final_move.reshape((self.n, self.n))


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = PotzAI()

    game.start_game()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memeory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory
            # print(game.get_state(), game.get_score())
            game.start_game()
            agent.n_games += 1
            agent.train_long_memeory()

            if score > record:
                record = score
                agent.model.save()

            # plot
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            print(f"Game {agent.n_games}, Score: {score}, Record: {record}, {mean_score=:.1f}\n")
            # if agent.n_games % 100 == 99:
            plot(plot_scores, plot_mean_scores)

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    # display.display(plt.gcf())
    # plt.clf()
    plt.title("Training...")
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    # plt.show()

if __name__ == "__main__":
    train()
