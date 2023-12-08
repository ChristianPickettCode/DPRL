import random
import copy
import numpy as np


class Connect4:
    def __init__(self):
        self.state = np.zeros((6, 7))
        self._drop_piece(self.state, 3, -1)
        # self._random_opponent_move(self.state)

    def step(self, action, state=None):
        if state is None:
            state = self.state

        valid_move = self._drop_piece(state, action, 1)

        if not valid_move:
            raise Exception("Invalid move")

        reward = self._check_winner(state)
        if reward != 0 or self._is_board_full(state):
            # print(reward, True)
            return reward, True

        self._random_opponent_move(state)
        reward = self._check_winner(state)

        game_over = False
        if reward != 0 or self._is_board_full(state):
            game_over = True
        # print(reward, game_over)
        return reward, game_over

    @classmethod
    def check_terminal(cls, state):
        reward = cls._check_winner(state)
        if reward != 0 or cls._is_board_full(state):
            return True
        return False

    @classmethod
    def get_legal_moves(cls, state):
        # Check if top most cell in that column is empty
        return [col for col in range(7) if state[0][col] == 0]

    def random_rollout(self, rollout_state=None):
        if rollout_state is None:
            rollout_state = self.state.copy()
        if self._is_board_full(rollout_state):
            return 0
        # Random rollout policy
        while True:
            next_move = random.sample(self.get_legal_moves(rollout_state), 1)[0]
            reward, done = self.step(next_move, rollout_state)
            if done:
                return reward

    def reset(self):
        self.state = np.zeros((6, 7))
        self._random_opponent_move(self.state)

    @classmethod
    def _random_opponent_move(cls, state):
        # Random agent's turn
        valid_move = False
        while not valid_move:
            column = random.randint(0, 6)
            valid_move = cls._drop_piece(state, column, -1)

    @classmethod
    def _drop_piece(cls, state, column, current_player):
        # Check for valid column
        if not 0 <= column < 7 or state[0][column] != 0:
            return False  # Invalid move

        # Find the lowest empty space in the column
        for row in range(5, -1, -1):
            if state[row][column] == 0:
                state[row][column] = current_player
                return True
        return False

    @classmethod
    def _is_board_full(cls, state):
        # Check if the board is full
        return all(state[0][col] != 0 for col in range(7))

    @classmethod
    def _check_winner(cls, state):
        # Check horizontal, vertical and diagonal lines for a win
        for row in range(6):
            for col in range(7):
                if state[row][col] != 0:
                    if (
                        cls._check_line(state, row, col, 1, 0)
                        or cls._check_line(state, row, col, 0, 1)
                        or cls._check_line(state, row, col, 1, 1)
                        or cls._check_line(state, row, col, 1, -1)
                    ):
                        return state[row][col]
        return 0

    @classmethod
    def _check_line(cls, state, start_row, start_col, d_row, d_col):
        # Check a line of 4 pieces starting from (start_row, start_col) in direction (d_row, d_col)
        for i in range(1, 4):
            r = start_row + d_row * i
            c = start_col + d_col * i
            if (
                not (0 <= r < 6 and 0 <= c < 7)
                or state[r][c] != state[start_row][start_col]
            ):
                return False
        return True
