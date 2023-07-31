import chess
import os
import torch
from MCTS import UCT_search, do_decode_n_move_pieces
from alpha_net import ChessNet as cnet
import cProfile
import torch.quantization


checkpoint=torch.load("model_data/current_net_trained_iter1.pth.tar")
bestmodel=cnet()
bestmodel.load_state_dict(checkpoint['state_dict'])
bestmodel.cuda()
bestmodel.eval()
# Quantize the model using dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(bestmodel, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
quantized_model.eval()
current_board = chess.Board()
current_board.set_fen("4R3/3kprPp/1p2b1N1/1nPNppb1/2P1p2Q/B1PP3P/2Kp1P2/5r2 w")
#best_move,_ = UCT_search(current_board, 400, bestmodel)
best_move,_ = cProfile.run("UCT_search(current_board, 400, bestmodel)")
move = do_decode_n_move_pieces(current_board, best_move)
print(move)