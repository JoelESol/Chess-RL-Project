import pickle
import os
import collections
import numpy as np
import math
import encoder_decoder as ed
import copy
import torch
import torch.multiprocessing as mp
from alpha_net import ChessNet
import datetime
import chess
import json
import random


class UCTNode():
    def __init__(self, board, move, parent=None):
        self.board = board  # state s
        self.move = move  # action index
        self.is_expanded = False
        self.parent = parent
        self.children = {}
        self.child_priors = np.zeros([4672], dtype=np.float32)
        self.child_total_value = np.zeros([4672], dtype=np.float32)
        self.child_number_visits = np.zeros([4672], dtype=np.float32)
        self.action_idxes = []

    @property
    def number_visits(self):
        return self.parent.child_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value):
        self.parent.child_number_visits[self.move] = value

    @property
    def total_value(self):
        return self.parent.child_total_value[self.move]

    @total_value.setter
    def total_value(self, value):
        self.parent.child_total_value[self.move] = value

    def child_Q(self):
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self):
        return math.sqrt(self.number_visits) * (abs(self.child_priors) / (1 + self.child_number_visits))

    def best_child(self):
        if self.action_idxes != []:
            bestmove = self.child_Q() + self.child_U()
            bestmove = self.action_idxes[np.argmax(bestmove[self.action_idxes])]
        else:
            bestmove = np.argmax(self.child_Q() + self.child_U())
        return bestmove

    def select_leaf(self):
        current = self
        while current.is_expanded:
            bestmove = current.best_child()
            current = current.maybe_add_child(bestmove)
        return current

    def add_dirichlet_noise(self, action_idxs, child_priors):
        valid_child_priors = child_priors[action_idxs]
        valid_child_priors = 0.85 * valid_child_priors + 0.15 * np.random.dirichlet(
            np.zeros([len(valid_child_priors)], dtype=np.float32) + 0.3)
        child_priors[action_idxs] = valid_child_priors
        return child_priors

    def expand(self, child_priors):
        self.is_expanded = True
        action_idxs = [];
        c_p = child_priors
        for action in list(self.board.legal_moves):
            if action != []:
                action_idxs.append(ed.encode_action(self.board, action))
        if action_idxs == []:
            self.is_expanded = False
        self.action_idxes = action_idxs
        for i in range(len(child_priors)):  # mask all illegal actions
            if i not in action_idxs:
                c_p[i] = 0.0000000
        if self.parent.parent == None:  # add dirichlet noise to child priors in root node
            c_p = self.add_dirichlet_noise(action_idxs, c_p)
        self.child_priors = c_p

    def decode_n_move_pieces(self, board, enmove):
        move = ed.decode_action(board, enmove)
        board.push(move)
        return board

    def maybe_add_child(self, move):
        if move not in self.children:
            board2 = copy.deepcopy(self.board)
            board2 = self.decode_n_move_pieces(board2, move)
            self.children[move] = UCTNode(board2, move, parent=self)
        return self.children[move]

    def backup(self, value_estimate: float):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            if current.board.turn == chess.BLACK:
                current.total_value += (1 * value_estimate)
            elif current.board.turn == chess.WHITE:
                current.total_value += (-1 * value_estimate)
            current = current.parent


class DummyNode(object):
    def __init__(self):
        self.parent = None
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)


def UCT_search(game_state, num_reads, net):
    root = UCTNode(game_state, move=None, parent=DummyNode())
    for i in range(num_reads):
        leaf = root.select_leaf()
        encoded_s = ed.encode_board(leaf.board)
        encoded_s = torch.from_numpy(encoded_s).float().cuda()
        child_priors, value_estimate = net(encoded_s)
        child_priors = child_priors.detach().cpu().numpy().reshape(-1);
        value_estimate = value_estimate.item()
        if leaf.board.is_game_over():
            leaf.backup(value_estimate);
            continue
        leaf.expand(child_priors)
        leaf.backup(value_estimate)
    return np.argmax(root.child_number_visits), root


def do_decode_n_move_pieces(board, enmove):
    move = ed.decode_action(board, enmove)
    return move


def get_policy(root):
    policy = np.zeros([4672], dtype=np.float32)
    for idx in np.where(root.child_number_visits != 0)[0]:
        policy[idx] = root.child_number_visits[idx] / root.child_number_visits.sum()
    return policy


def save_as_pickle(filename, data):
    completeName = os.path.join("./datasets/iter5/", filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)


def load_pickle(filename):
    completeName = os.path.join("./datasets/", filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data


def MCTS_self_play(chessnet, num_games, cpu):
    openings={}
    with open('opening_books/opening_fens.json', 'r') as openfile:
        openings = json.load(openfile)

    for idxx in range(0, num_games):
        board = chess.Board()
        fen=random.choice(list(openings.values()))
        #print(fen)
        board.set_fen(fen)
        dataset = []
        states = []
        value = 0
        while not board.is_game_over() and board.fullmove_number<125:
            states.append(copy.deepcopy(board))
            board_state = copy.deepcopy(ed.encode_board(board))
            best_move, root = UCT_search(board, 400, chessnet)
            move = do_decode_n_move_pieces(board, best_move)
            board.push(move)
            policy = get_policy(root)
            dataset.append([board_state, policy])
            #print(board, board.fullmove_number)
            if board.is_checkmate():
                print("checkmate")
                if board.result() == "1-0":
                    value = (1 + 0.2 * (100 - board.fullmove_number) / 100) / 1.2
                    print("white wins")
                else:
                    value = (-1 - 0.2 * (100 - board.fullmove_number) / 100) / 1.2
                    print("black wins")
        dataset_p = []
        for idx, data, in enumerate(dataset):
            s, p = data
            if idx == 0:
                dataset_p.append([s, p, 0])
            else:
                dataset_p.append([s, p, value])
        del dataset
        save_as_pickle("dataset_cpu%i_%i_%s" % (cpu, idxx, datetime.datetime.today().strftime("%Y-%m-%d")), dataset_p)


if __name__ == "__main__":

    net_to_play = "current_net_trained_iter0.pth.tar"
    mp.set_start_method("spawn", force=True)
    net = ChessNet()
    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()
    net.share_memory()
    net.eval()
    torch.save({'state_dict': net.state_dict()}, os.path.join("./model_data/", "current_net_trained_iter0.pth.tar"))
    current_net_filename = os.path.join("./model_data/", net_to_play)
    print("saved")
    checkpoint = torch.load(current_net_filename)
    net.load_state_dict(checkpoint['state_dict'])
    processes = []
    for i in range(36):
        p = mp.Process(target=MCTS_self_play, args=(net, 20, i))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
