import os.path
import torch
import numpy as np
from alpha_net import ChessNet as cnet
import chess
import encoder_decoder as ed
import copy
from MCTS import UCT_search, do_decode_n_move_pieces
import pickle
import torch.multiprocessing as mp


def save_as_pickle(filename, data):
    completeName = os.path.join("./evaluator_data/", \
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)


class arena():
    def __init__(self, current_chessnet, best_chessnet):
        self.current = current_chessnet
        self.best = best_chessnet

    def play_round(self):
        if np.random.uniform(0, 1) <= 0.5:
            white = self.current;
            black = self.best;
            w = "current";
            b = "best"
        else:
            white = self.best;
            black = self.current;
            w = "best";
            b = "current"
        current_board = chess.Board()
        checkmate = False
        states = [];
        dataset = []
        value = 0
        while not current_board.is_game_over():
            draw_counter = 0
            states.append(current_board)
            board_state = ed.encode_board(current_board)
            dataset.append(board_state)
            if current_board.player == 0:
                best_move, _ = UCT_search(current_board, 777, white)
            elif current_board.player == 1:
                best_move, _ = UCT_search(current_board, 777, black)
            move = do_decode_n_move_pieces(current_board, best_move)  # decode move and move piece(s)
            current_board.push(move)
            print(current_board, current_board.fullmove_number);
            print(" ")
            if current_board.is_checkmate():  # checkmate
                if current_board.result() == "1-0":
                    value = (1 + 0.2 * ((100 - current_board.fullmove_number) / 100) / 1.2)
                else:
                    value = (-1 - 0.2 * ((100 - current_board.fullmove_number) / 100) / 1.2)
        dataset.append(value)
        if value < 0:
            return b, dataset
        elif value > 0:
            return w, dataset
        else:
            return None, dataset

    def evaluate(self, num_games, cpu):
        current_wins = 0
        for i in range(num_games):
            winner, dataset = self.play_round();
            print("%s wins!" % winner)
            dataset.append(winner)
            if winner == "current":
                current_wins += 1
            save_as_pickle("evaluate_net_dataset_cpu%i_%i" % (cpu, i), dataset)
        print("Current_net wins ratio: %.3f" % current_wins / num_games)
        # if current_wins/num_games > 0.55: # saves current net as best net if it wins > 55 % games
        #    torch.save({'state_dict': self.current.state_dict()}, os.path.join("./model_data/",\
        #                                "best_net.pth.tar"))


def fork_process(arena_obj, num_games, cpu):  # make arena picklable
    arena_obj.evaluate(num_games, cpu)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    current_net = "current_net.pth.tar";
    best_net = "current_net_trained.pth.tar"
    current_net_filename = os.path.join("./model_data/", \
                                        current_net)
    best_net_filename = os.path.join("./model_data/", \
                                     best_net)
    current_chessnet = cnet()
    best_chessnet = cnet()
    checkpoint = torch.load(current_net_filename)
    current_chessnet.load_state_dict(checkpoint['state_dict'])
    checkpoint = torch.load(best_net_filename)
    best_chessnet.load_state_dict(checkpoint['state_dict'])
    cuda = torch.cuda.is_available()
    if cuda:
        current_chessnet.cuda()
        best_chessnet.cuda()
    current_chessnet.eval();
    best_chessnet.eval()
    current_chessnet.share_memory();
    best_chessnet.share_memory()

    processes = []
    for i in range(6):
        p = mp.Process(target=fork_process, args=(arena(current_chessnet, best_chessnet), 50, i))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
