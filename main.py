import chess
import encoder_decoder as ed
board=chess.Board()
moves=list(board.legal_moves)
print(list(board.legal_moves))
board.push(moves[1])
move=ed.decode_action(board, 646)
print(move)

