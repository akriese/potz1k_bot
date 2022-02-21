import numpy as np
import random

class PotzAI:
    def __init__(self, n=3) -> None:
        self.n = n
        self.goal = int(10**self.n)
        self.multipliers = np.power(10, np.flip(np.arange(self.n)))
        self.reset()

    def reset(self):
        self.matrix = np.zeros((self.n, self.n))
        self.rolls_left = self.n * self.n
        self.last_dice = 0
        self.score = 0

    def start_game(self):
        self.reset()
        self.next_dice()

    def get_state(self):
        state = np.hstack((self.matrix.flatten(), self.last_dice))
        return state

    def play_step(self, action=None):
        """
            'action' is meant to be an N*N array with exactly one 1, else 0s
            to denote the placement of the rolled dices number
        """
        old_score = self.get_score(self.matrix)
        # print(action)
        if isinstance(action, tuple): # for human input
            self.matrix[action[0], action[1]] = self.last_dice
        else:
            if action.sum() != 1:
                print(f"Wrong action: {action}")
            if np.sum(self.matrix * action) != 0:
                # already taken value was chosen, penalty
                return -1000, True, -1000
            self.matrix += action * self.last_dice

        new_score = self.get_score(self.matrix)
        reward = new_score - old_score

        if self.rolls_left == 0:
            # print(self.matrix)
            return reward, True, new_score

        self.next_dice()

        return reward, False, new_score

    def next_dice(self):
        self.last_dice = random.randint(1, 6)
        self.rolls_left -= 1
        return self.last_dice

    def get_score(self, matrix=None):
        if matrix is None:
            matrix = self.matrix
        current_sum = self.get_sum(matrix)
        score = self.goal - np.abs(self.goal - current_sum)
        # diagonal bonus
        if np.equal.reduce(matrix.diagonal()):
            score += matrix[0, 0] * 10
        if np.equal.reduce(np.fliplr(matrix).diagonal()):
            score += matrix[self.n-1, 0] * 10

        return score

    def get_sum(self, matrix=None):
        if matrix is None:
            matrix = self.matrix
        return (matrix * self.multipliers).sum()

