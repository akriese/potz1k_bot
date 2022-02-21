from collections import deque
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns
import sys
import torch

from game import PotzAI
from model import QTrainer, LinearQNet

MAX_MEMORY = 100_000
BATCH_SIZE = 128
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
            # arg_max = torch.argmax(prediction[0][possible_moves]).item()
            # move = possible_moves[arg_max]
            move = torch.argmax(prediction[0]).item()
            final_move[move] = 1
            if final_move.sum() != 1:
                print(f"{move=}, {state=}, {prediction=}")

        return final_move.reshape((self.n, self.n))


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    N = 3
    agent = Agent(n=N)
    game = PotzAI(n=N)
    first_choices = np.zeros((N*N, N, N))

    game.start_game()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)
        first_choices[N*N - game.rolls_left - 1] += final_move

        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memeory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory
            # print(game.get_state(), game.get_score())
            game.start_game()
            agent.n_games += 1

            if score > record:
                record = score
                agent.model.save()

            # plot
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            last_50_scores = np.mean(plot_scores[-50:])
            plot_mean_scores.append(mean_score)
            print(f"Game {agent.n_games}, Score: {score}, ",
                f"Record: {record}, Last50: {last_50_scores:.1f}")
            if agent.n_games % 500 == 0:
                _, axs = plt.subplots(nrows=N, ncols=N)
                for i, ax_row in enumerate(axs):
                    for j, ax in enumerate(ax_row):
                        move = i*N+j
                        ax.set_title(f"Move {move+1}")
                        sns.heatmap(data=first_choices[move], ax=ax)

                plt.show()
            # plot(plot_scores, plot_mean_scores)

            # print("training LT memory")
            agent.train_long_memeory()

def play_example(agent: Agent, game: PotzAI):
    game.start_game()
    n = game.n
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        _, done, _ = game.play_step(final_move)
        status_new = agent.get_state(game)

        print(f"{status_new[:-1].reshape((n, n))}")


        if done:
            print(f"Bot scored {game.get_score()}")
            choice = input("Do you want the bot to play another game?")
            if choice in ['y', 'Y', '']:
                print(f"####### Starting new game... ########")
                game.start_game()
            else:
                break
        else:
            print(f"Next dice: {status_new[-1]}")

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
    if sys.argv[1] == 'train':
        train()
    else:
        if len(sys.argv) >= 3:
            fname = sys.argv[2]
        agent = Agent(n=3)
        agent.model.load()
        game = PotzAI(n=3)
        play_example(agent, game)


