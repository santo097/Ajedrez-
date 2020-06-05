import numpy as np
import copy
import chess
from Reinforcement_Learning.Monte_Carlo_Search_Tree.self_play import start

class MCTS(object):
    def __init__(self, Neural_Network):

        self.root = Node(None, 1.0)
        self.Network = Neural_Network

    def iteration_of_MCTS(self, state, last = 0):
        while(1):
            last, answer = self.root.is_leaf(last)
            if answer or last:
                break

            move, node = self.root.traverse_tree(5)
            move = chess.Move.from_uci(str(move))
            state.push(action)
        probability, leaf_value = self.Network(state)

        s = start(state)
        end, winner = s.results(state, state.result())

        if not end:
            self.root.leaf_expansion(probability)
        else:
            if winner == -1:
                leaf_value = 0.0
            else:
                leaf_value = 1.0

        self.root.backpropagation(-leaf_value)

    def move_probabilities(self, state, temperature=1e-3):

        for n in range(10000):
            #Deepcopy is very slow, but I found it works the best
            state_copy = copy.deepcopy(state)
            self.iteration_of_MCTS(state_copy)
        # calc the move probabilities based on visit counts at the root node

        act_visits = [(act, node.N) for act, node in self.root.children.items()]
        moves, visits = zip(*act_visits)

        act_probs = softmax(1.0/temperature * np.log(np.array(visits) + 1e-10))

        return moves, act_probs

    def update_tree(self, last_move):

        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = Node(None, 1.0)

class agent_MCTS(object):

    def __init__(self, policy_value_function, training=0):
        self.mcts = MCTS(policy_value_function)
        self.training = training

    def choose_move(self, board, temperature=1e-3, probability=0):

        legal_moves = list(board.legal_moves)
        probability_for_move = np.zeros( len(legal_moves) )
        moves, probs = self.mcts.move_probabilities(board, temperature)

        index = 0
        for i in list(moves):
            probability_for_move[index] = probs[index]
            index += 1

        if self.training:
            move = np.random.choice( moves, p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs))) )
            self.mcts.update_tree(move)
        else:
            move = np.random.choice(moves, p=probs)
            self.mcts.update_tree(-1)

        if probability:
            return move, probability_for_move
        else:
            return move

    def reset_player(self):
        self.mcts.update_tree(-1)

    def set_player(self, pl):
        self.player = pl

class Node(object):

    def __init__(self, parent, probability):
        self.parent = parent

        self.states = {}
        self.children = {}

        self.N = 0
        self.Q = 0
        self.U = 0
        self.P = probability

    def traverse_tree(self, Cpuct):

        return max(self.children.items(), key=lambda act_node: act_node[1].node_value(Cpuct))

    def leaf_expansion(self, move_history):

        for move, probability in move_history:
            if move not in self.children:
                self.children[move] = Node(self, probability)

    def node_value(self, Cpuct):
        self.U = (Cpuct * self.P * np.sqrt(self.parent.N) / (1 + self.N))
        return self.Q + self.U

    def update_leaf_node(self, leaf_value):
        # Count visit.
        self.N += 1
        # Update Q, a running average of values for all visits.
        self.Q += 1.0*(leaf_value - self.Q) / self.N

    def backpropagation(self, leaf_value):

        # If it is not root, this node's parent should be updated first.
        if self.parent:
            self.parent.backpropagation(-leaf_value)

        self.update_leaf_node(leaf_value)

    def is_leaf(self,last):

        if last == self.children:
            return True, 0
        return self.children, self.children == {}

    def is_root(self):
        return self.parent is None

def softmax(x):
    probabilities = np.exp(x - np.max(x))
    probabilities /= np.sum(probabilities)
    return probabilities
