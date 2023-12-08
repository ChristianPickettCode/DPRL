import numpy as np
from connect4 import Connect4
from mcts import MCTS

game = Connect4()
# start_state = np.array(
#     [
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0, 0, 0],
#         [0, 1, 0, -1, -1, 0, -1],
#     ]
# )
start_state = np.array(
    [
        [0, 1, -1, -1, 1, -1, 0],
        [0, 1, 1, -1, -1, 1, 0],
        [0, -1, -1, -1, 1, -1, 0],
        [0, 1, 1, 1, -1, 1, 0],
        [0, 1, -1, -1, 1, -1, 0],
        [-1, 1, 1, 1, -1, -1, 1],
    ]
)
game.state = start_state
mcts = MCTS(game)
best_action = mcts.get_best_action(maxiter=2048)
print(
    f"best action for start state: {best_action.action}, expected val: {mcts.root.vval:.4f}"
)
print(game.state)
n = 50
wins = 0
losses = 0
res = 0
draws = 0

reward = None
for _ in range(n):
    game.state = start_state.copy()
    # game.reset()
    done = False
    while not done:
        best_action = mcts.get_best_action(maxiter=128)
        reward, done = game.step(best_action.action)
        if done:
            if reward > 0:
                wins += 1
            elif reward < 0:
                losses += 1
            else:
                draws += 1
            res += reward

print(f"actual value: {res / n}")
print(f"wins {wins}, losses: {losses}, draws: {draws}")
