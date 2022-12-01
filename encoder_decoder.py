import numpy as np
import chess
import time
import random


def encode_board(board):
    encoded = np.zeros([13, 8, 8]).astype(int)
    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.WHITE):
            idx = np.unravel_index(square, (8, 8))
            encoded[piece - 1][7 - idx[0]][idx[1]] = 1
        for square in board.pieces(piece, chess.BLACK):
            idx = np.unravel_index(square, (8, 8))
            encoded[piece + 5][7 - idx[0]][idx[1]] = 1
    if board.turn == chess.WHITE:
        encoded[12][0][:] = 1
    if bool(board.castling_rights & chess.BB_H1):
        encoded[12][1][:] = 1
    if bool(board.castling_rights & chess.BB_A1):
        encoded[12][2][:] = 1
    if bool(board.castling_rights & chess.BB_A8):
        encoded[12][3][:] = 1
    if bool(board.castling_rights & chess.BB_H8):
        encoded[12][4][:] = 1
    if bool(board.is_variant_draw()):
        encoded[12][5][:] = 1
    return encoded


def encode_action(board, move):
    init_square = move.from_square
    final_square = move.to_square
    initial_pos = [chess.square_file(init_square), chess.square_rank(init_square)]
    final_pos = [chess.square_file(final_square), chess.square_rank(final_square)]
    encoded = np.zeros([8, 8, 73]).astype(int)
    underpromote = move.promotion
    i, j = initial_pos;
    x, y = final_pos;
    dx, dy = x - i, y - j;
    piece = board.piece_at(init_square)
    piece = piece.symbol()
    if piece in ["R", "B", "Q", "K", "P", "r", "b", "q", "k", "p"] and underpromote in [None, 5]:
        if dx != 0 and dy == 0:  # N-S idx 0-13
            if dx < 0:
                idx = 7 + dx
            elif dx > 0:
                idx = 6 + dx
        if dx == 0 and dy != 0:  # E-W idx 14-27
            if dy < 0:
                idx = 21 + dy
            elif dy > 0:
                idx = 20 + dy
        if dx == dy:  # NW-SE idx 28-41
            if dx < 0:
                idx = 35 + dx
            if dx > 0:
                idx = 34 + dx
        if dx == -dy:  # NE-SW idx 42-55
            if dx < 0:
                idx = 49 + dx
            if dx > 0:
                idx = 48 + dx
    if piece in ["n", "N"]:  # knight moves 56-63
        if (x, y) == (i + 2, j - 1):
            idx = 56
        if (x, y) == (i + 2, j + 1):
            idx = 57
        if (x, y) == (i + 1, j - 2):
            idx = 58
        if (x, y) == (i - 1, j - 2):
            idx = 59
        if (x, y) == (i - 2, j + 1):
            idx = 60
        if (x, y) == (i - 2, j - 1):
            idx = 61
        if (x, y) == (i - 1, j + 2):
            idx = 62
        if (x, y) == (i + 1, j + 2):
            idx = 63
    if piece in ["p", "P"] and (y == 0 or y == 7) and underpromote != None:
        if abs(dy) == 1 and dx == 0:
            if underpromote == 4:  # ROOK
                idx = 64
            if underpromote == 2:  # KNIGHT
                idx = 65
            if underpromote == 3:  # BISHOP
                idx = 66
        if abs(dy) == 1 and dx == -1:
            if underpromote == 4:  # ROOK
                idx = 67
            if underpromote == 2:  # KNIGHT
                idx = 68
            if underpromote == 3:  # BISHOP
                idx = 69
        if abs(dy) == 1 and dx == 1:
            if underpromote == 4:  # ROOK
                idx = 70
            if underpromote == 2:  # KNIGHT
                idx = 71
            if underpromote == 3:  # BISHOP
                idx = 72
    encoded[i, j, idx] = 1
    encoded = encoded.reshape(-1);
    encoded = np.where(encoded == 1)[0][0]
    return encoded


def decode_action(board, encoded):
    encoded_a = np.zeros([4672]);
    encoded_a[encoded] = 1;
    encoded_a = encoded_a.reshape(8, 8, 73)
    a, b, c = np.where(encoded_a == 1);
    for pos in zip(a, b, c):
        i, j, k = pos
        initial_pos = (i, j)
        promoted = None
        if 0 <= k <= 13:  # North-south
            dy = 0
            if k < 7:
                dx = k - 7
            else:
                dx = k - 6
            final_pos = (i + dx, j + dy)
        elif 14 <= k <= 27:  # east-west
            dx = 0
            if k < 21:
                dy = k - 21
            else:
                dy = k - 20
            final_pos = (i + dx, j + dy)
        elif 28 <= k <= 41:  # NW-SE
            if k < 35:
                dy = k - 35
            else:
                dy = k - 34
            dx = dy
            final_pos = (i + dx, j + dy)
        elif 42 <= k <= 55:  # NE-SW
            if k < 49:
                dx = k - 49
            else:
                dx = k - 48
            dy = -dx
            final_pos = (i + dx, j + dy)
        elif 56 <= k <= 63:  # Knight Moves
            if k == 56:
                final_pos = (i + 2, j - 1)
            elif k == 57:
                final_pos = (i + 2, j + 1)
            elif k == 58:
                final_pos = (i + 1, j - 2)
            elif k == 59:
                final_pos = (i - 1, j - 2)
            elif k == 60:
                final_pos = (i - 2, j + 1)
            elif k == 61:
                final_pos = (i - 2, j - 1)
            elif k == 62:
                final_pos = (i - 1, j + 2)
            elif k == 63:
                final_pos = (i + 1, j + 2)

        else:
            if k == 64:
                if board.turn == chess.WHITE:
                    final_pos = (i, j + 1)
                    promoted = 4  # ROOK
                else:
                    final_pos = (i, j - 1)
                    promoted = 4  # rook
            if k == 65:
                if board.turn == chess.WHITE:
                    final_pos = (i, j + 1)
                    promoted = 2  # KNIGHT
                else:
                    final_pos = (i, j - 1)
                    promoted = 2  # knight
            if k == 66:
                if board.turn == chess.WHITE:
                    final_pos = (i, j + 1)
                    promoted = 3  # BISHOP
                else:
                    final_pos = (i, j - 1)
                    promoted = 3  # bishop
            if k == 67:
                if board.turn == chess.WHITE:
                    final_pos = (i - 1, j + 1)
                    promoted = 4  # ROOK
                else:
                    final_pos = (i - 1, j - 1)
                    promoted = 4  # rook
            if k == 68:
                if board.turn == chess.WHITE:
                    final_pos = (i - 1, j + 1)
                    promoted = 2  # KNIGHT
                else:
                    final_pos = (i - 1, j - 1)
                    promoted = 2  # knight
            if k == 69:
                if board.turn == chess.WHITE:
                    final_pos = (i - 1, j + 1)
                    promoted = 3  # BISHOP
                else:
                    final_pos = (i - 1, j - 1)
                    promoted = 3  # bishop
            if k == 70:
                if board.turn == chess.WHITE:
                    final_pos = (i + 1, j + 1)
                    promoted = 4  # ROOK
                else:
                    final_pos = (i + 1, j - 1)
                    promoted = 4  # rook
            if k == 71:
                if board.turn == chess.WHITE:
                    final_pos = (i + 1, j + 1)
                    promoted = 2  # KNIGHT
                else:
                    final_pos = (i + 1, j - 1)
                    promoted = 2  # knight
            if k == 72:
                if board.turn == chess.WHITE:
                    final_pos = (i + 1, j + 1)
                    promoted = 3  # BISHOP
                else:
                    final_pos = (i + 1, j - 1)
                    promoted = 3  # bishop

        piece = board.piece_at(chess.square(initial_pos[0], initial_pos[1]))
        piece = piece.symbol()
        if piece in ["P", "p"] and final_pos[1] in [0, 7] and promoted == None:
            if board.turn == chess.WHITE:
                promoted = 5  # QUEEN
            else:
                promoted = 5  # queen

    initsquare = initial_pos[1] * 8 + initial_pos[0]
    finalsquare = final_pos[1] * 8 + final_pos[0]
    move = chess.Move(initsquare, finalsquare, promoted)
    return move
