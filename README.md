# Potz 1000 RL

This is a reinforcement learning approach to playing the simple game 'Potz 1000'.

## The game
The game is played by rolling a dice nine times and writing down the number each time.
There are usually 3x3 slots to put the numbers in. Each row is considered a three digit number
and all three numbers are added up in the end to calculate the final score.

The goal of the game is to get this sum as close as possible to 1000, hence the name.
The score is the difference between 1000 and the sum. The lower the score, the better.
Additionally, one can get a bonus by having a diagonal with three times the same number.
This number is multiplied by ten and subtracted from the original game score.

Examples:
```
  3 4 3
+ 2 1 6
+ 4 1 1
-------
1 0 7 0  ==> Score: abs(1070 - 1000) = 70

  4 4 2
+ 2 2 5
+ 2 3 1
-------
  8 9 8  ==> Score: abs(898 - 1000) - 2*10 = 82
```

## The algorithm
The game is implemented in python and so are the agent and Deep Neural network (pytorch).
The approach is highly inspired by [python-engineer's](https://github.com/python-engineer)
[snake-ai-pytorch](https://github.com/python-engineer/snake-ai-pytorch) project and thus uses
Reinforcement and Deep Q learning.


## Usage
The training of the agent can be done by invoking
```bash
python agent.py train
```
which will train an agent infinitely.

To run the trained model for a single game,
```bash
python agent.py play [model_name]
```
can be used. This automatically uses the saved model. Alternatively, `model_name`
can be specified to chose a different saved model.


## Requirements
- pytorch
- matplotlib
- seaborn
- numpy


## Thoughts
Potz 1000 is a game that is heavily influenced by randomness (rolling the dice).
Thus, the learning of the agent is not very effective.
Also, the reward system for the reinforcement learning part of the agent is currently not optimal,
as rewards are given for each step separately. This is not realistic, as consecutive play steps influence each other leading to the need of connected rewards. This has yet to be implemented.





