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
#from other_players import Flatline, StockFish



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
        current_board.set_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1")
        game = chess.pgn.Game()
        game.headers["White"] = w
        game.headers["Black"] = b
        game.setup(current_board)
        node = game
        value=0
        while not current_board.is_game_over():
            if current_board.turn == chess.WHITE:
                best_move, _ = UCT_search(current_board, 400, white)
            elif current_board.turn == chess.BLACK:
                best_move, _ = UCT_search(current_board, 400, black)
            move = do_decode_n_move_pieces(current_board, best_move)  # decode move and move piece(s)
            ucimove = move.uci()
            # print(ucimove)
            node = node.add_variation(chess.Move.from_uci(ucimove))
            current_board.push(move)
            #print(current_board, current_board.fullmove_number);
            #print(" ")
            if current_board.is_checkmate():  # checkmate
                if current_board.result() == "1-0":
                    value = (1 + 0.2 * (100 - current_board.fullmove_number) / 100) / 1.2
                else:
                    value = (-1 - 0.2 * (100 - current_board.fullmove_number) / 100) / 1.2
        print(game)
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
            if winner == "current":
                current_wins += 1
        print("Current_net wins ratio: %.3f" % current_wins / num_games)


def fork_process(arena_obj, num_games, cpu):  # make arena picklable
    arena_obj.evaluate(num_games, cpu)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    current_net = "current_net.pth.tar";
    best_net = "current_net_trained_iter9.pth.tar"
    current_net_filename = os.path.join("./resmodels/", \
                                        current_net)
    best_net_filename = os.path.join("./resmodels/", \
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
        p = mp.Process(target=fork_process, args=(arena(current_chessnet, best_chessnet), 1, i))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
