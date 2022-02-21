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
        self.optimal_score = self.goal

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

        # reward = new_score - old_score
        old_opt_score = self.optimal_score
        self.optimal_score = self.best_possible_score()
        reward = (new_score - old_score) / (old_opt_score - self.optimal_score + 1)


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
        if np.all(matrix.diagonal() == matrix[0, 0]):
            score += matrix[0, 0] * 10
        if np.all(np.fliplr(matrix).diagonal() == matrix[self.n-1, 0]):
            score += matrix[self.n-1, 0] * 10

        return score

    def get_sum(self, matrix=None):
        if matrix is None:
            matrix = self.matrix
        return (matrix * self.multipliers).sum()

    def best_possible_score(self, matrix=None):
        # determine best score without thinking about bonuses
        if matrix is None:
            matrix = self.matrix
        n = matrix.shape[0]
        def rec_search(spots, diff):
            best_dist = []
            for i, s in enumerate(spots):
                if s == 0:
                    continue
                new_spots = spots.copy()
                new_spots[i] -= 1
                for d in range(1, 7):
                    new_diff = diff - d*10**(n-i-1)
                    if new_diff < 0:
                        best_dist.append(abs(new_diff - set_all_ones(new_spots)))
                        break

                    # print(diff, d, i, new_diff, new_spots)
                    best_dist.append(rec_search(new_spots, new_diff))
                break
            if not best_dist:
                return diff
            return min(best_dist)

        def set_all_ones(spots):
            return np.sum(self.multipliers * spots)

        current_sum = self.get_sum(matrix)
        mask = matrix == 0
        if current_sum == 0 or np.sum(mask) == n*n-1:
            return self.goal
        if np.sum(mask) == n*n:
            return abs(self.goal-current_sum)

        available_spots = np.zeros(n)
        for i in range(n):
            available_spots[i] = mask[:, i].sum()

        if current_sum > self.goal:
            # minimize additional points towards goal
            # put lowest number into all available spots
            best_score = current_sum + np.sum(available_spots * self.multipliers)
        else:
            # maximize additional points towards goal
            # here, we need some greedy algorithm
            diff = self.goal - current_sum
            best_diff = rec_search(available_spots, diff)
            # print(self.goal, diff, best_diff)
            best_score = self.goal - best_diff

        return best_score

