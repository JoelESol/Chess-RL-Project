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
import chess.pgn


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
        game=chess.pgn.Game()
        game.headers["White"]=w
        game.headers["Black"]=b
        game.setup(current_board)
        node=game
        while not current_board.is_game_over():
            if current_board.turn == chess.WHITE:
                best_move, _ = UCT_search(current_board, 10, white)
            elif current_board.turn == chess.BLACK:
                best_move, _ = UCT_search(current_board, 10, black)
            move = do_decode_n_move_pieces(current_board, best_move)  # decode move and move piece(s)
            ucimove=move.uci()
            #print(ucimove)
            node=node.add_variation(chess.Move.from_uci(ucimove))
            current_board.push(move)
        #print(game)
        if value < 0:
            return b
        elif value > 0:
            return w
        else:
            return None

    def evaluate(self, num_games, cpu):
        current_wins = 0
        for i in range(num_games):
            winner = self.play_round();
            print("%s wins!" % winner)
            if winner == "best":
                current_wins += 1
            #save_as_pickle("evaluate_net_dataset_cpu%i_%i" % (cpu, i), dataset)
        print(current_wins)
        # if current_wins/num_games > 0.55: # saves current net as best net if it wins > 55 % games
        #    torch.save({'state_dict': self.current.state_dict()}, os.path.join("./model_data/",\
        #                                "best_net.pth.tar"))


def fork_process(arena_obj, num_games, cpu):  # make arena picklable
    arena_obj.evaluate(num_games, cpu)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    current_net = "current_net_trained_iter2.pth.tar";
    best_net = "current_net_trained_iter3.pth.tar"
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
    for i in range(4):
        p = mp.Process(target=fork_process, args=(arena(current_chessnet, best_chessnet), 5, i))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
