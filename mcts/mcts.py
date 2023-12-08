import numpy as np
import math


class MCTS:
    def __init__(self, game):
        self.game = game

    def get_best_action(self, maxiter=64):
        state = self.game.state.copy()
        self.root = StateNode(None, state)

        for _ in range(maxiter):
            curnode = self._run_tree_search(self.root)
            reward = self._perform_rollout(curnode)
            self._backpropagate(curnode, reward)

        # print(self.root.vval)
        # for action in self.root.actions.values():
        #     print(f"{action.action} {action.qval:.2f}")
        return self._get_maxq_action(self.root), self.root.actions.values()

    def print_tree(self, node=None):
        if node is None:
            node = self.root
        print(f"====={node.id}======")
        print(self._state_to_np(node.state))
        print(node.cum_reward)
        for action in node.actions.values():
            print(action.action, action.qval)

        for action in node.actions.values():
            for child in action.children_states.values():
                self.print_tree(child)

    def _get_maxq_action(self, node):
        qval = -math.inf
        best_action = None
        for action in node.actions.values():
            if action.qval > qval:
                qval = action.qval
                best_action = action
        return best_action

    def _run_tree_search(self, node):
        cur_node = node
        done = self.game.check_terminal(self._state_to_np(cur_node.state))
        while not done:
            if len(cur_node.actions) == 0:
                cur_node_state_np = self._state_to_np(cur_node.state)
                cur_node.expand_node(self.game.get_legal_moves(cur_node_state_np))
            action_node = self._select_action(cur_node)
            new_state, done = self._get_next_state(cur_node.state, action_node.action)
            if new_state not in action_node.children_states:
                new_state_node = StateNode(action_node, new_state)
                action_node.children_states[new_state] = new_state_node
                return new_state_node
            cur_node = action_node.children_states[new_state]
        return cur_node

    def _perform_rollout(self, node):
        state_np = self._state_to_np(node.state)
        return self.game.random_rollout(state_np)

    def _backpropagate(self, state_node, reward):
        state_node.count += 1
        state_node.cum_reward += reward
        node = state_node
        action = node.parent
        while action is not None:
            node = action.parent
            action.count += 1
            action.cum_reward += reward
            node.count += 1
            node.cum_reward += reward
            action = node.parent

    # def _select_action(self, node):
    #     for action in node.actions.values():
    #         if action.count == 0:
    #             return action
    #     return random.sample(list(node.actions.values()), 1)[0]

    def _select_action(self, node):
        best_ucb = -math.inf
        best_action = None
        for action in node.actions.values():
            if action.count == 0:
                curucb = math.inf
            else:
                curucb = action.qval + math.sqrt(
                    2 * math.log(node.count) / action.count
                )
            if curucb > best_ucb:
                best_ucb = curucb
                best_action = action
        return best_action

    def _get_next_state(self, state, action):
        state_np = self._state_to_np(state)
        _, done = self.game.step(action, state_np)
        new_state = state_np.tobytes()
        return new_state, done

    def _state_to_np(self, state):
        state_np = np.frombuffer(state, dtype=int).copy()
        state_np = state_np.reshape(self.game.state.shape)
        return state_np


class StateNode:
    node_count = 0

    def __init__(self, parent, state):
        self.parent = parent
        self.state = state
        self.actions = {}
        self.count = 0
        self.cum_reward = 0
        self.id = StateNode.node_count
        StateNode.node_count += 1

    def expand_node(self, available_actions):
        for action in available_actions:
            self.actions[action] = ActionNode(self, action)

    @property
    def vval(self):
        if self.count == 0:
            return 0
        return self.cum_reward / self.count


class ActionNode:
    def __init__(self, parent, action):
        self.parent = parent
        self.action = action
        self.children_states = {}
        self.count = 0
        self.cum_reward = 0

    @property
    def qval(self):
        if self.count == 0:
            return 0
        return self.cum_reward / self.count
