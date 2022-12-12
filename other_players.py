import chess

def random_player(board):
    move = random.choice(list(board.legal_moves))
    return move.uci()

def Flatline(board):
    board2 = board

    pawntablew = [
        0, 0, 0, 0, 0, 0, 0, 0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5, 5, 10, 25, 25, 10, 5, 5,
        0, 0, 0, 20, 20, 0, 0, 0,
        5, -5, -10, 0, 0, -10, -5, 5,
        5, 10, 10, -20, -20, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0
    ]
    knightstablew = [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 10, 15, 15, 10, 5, -30,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50,
    ]
    bishoptablew = [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 5, 5, 10, 10, 5, 5, -10,
        -10, 0, 10, 10, 10, 10, 0, -10,
        -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 5, 0, 0, 0, 0, 5, -10,
        -20, -10, -10, -10, -10, -10, -10, -20,
    ]
    rooktablew = [
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, 10, 10, 10, 10, 5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        0, 0, 0, 5, 5, 0, 0, 0
    ]
    queentablew = [
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -5, 0, 5, 5, 5, 5, 0, -5,
        0, 0, 5, 5, 5, 5, 0, -5,
        -10, 5, 5, 5, 5, 5, 0, -10,
        -10, 0, 5, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20
    ]

    kingtablew = [
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        20, 20, 0, 0, 0, 0, 20, 20,
        20, 30, 10, 0, 0, 10, 30, 20
    ]

    pawntableb = [
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, -20, -20, 10, 10, 5,
        5, -5, -10, 0, 0, -10, -5, 5,
        0, 0, 0, 20, 20, 0, 0, 0,
        5, 5, 10, 25, 25, 10, 5, 5,
        10, 10, 20, 30, 30, 20, 10, 10,
        50, 50, 50, 50, 50, 50, 50, 50,
        0, 0, 0, 0, 0, 0, 0, 0
    ]
    knightstableb = [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -30, 5, 10, 15, 15, 10, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50,
    ]
    bishoptableb = [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 5, 0, 0, 0, 0, 5, -10,
        -10, 10, 10, 10, 10, 10, 10, -10,
        -10, 0, 10, 10, 10, 10, 0, -10,
        -10, 5, 5, 10, 10, 5, 5, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -20, -10, -10, -10, -10, -10, -10, -20,
    ]
    rooktableb = [
        0, 0, 0, 5, 5, 0, 0, 0,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        5, 10, 10, 10, 10, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0
    ]
    queentableb = [
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 0, 5, 0, 0, 0, 0, -10,
        -10, 5, 5, 5, 5, 5, 0, -10,
        0, 0, 5, 5, 5, 5, 0, -5,
        -5, 0, 5, 5, 5, 5, 0, -5,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20
    ]

    kingtableb = [
        20, 30, 10, 0, 0, 10, 30, 20,
        20, 20, 0, 0, 0, 0, 20, 20,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -10, -20, -20, -20, -20, -20, -20, -10,
    ]
    tablesw = [pawntablew,knightstablew,bishoptablew,rooktablew,queentablew,kingtablew]
    tablesb = [pawntableb,knightstableb,bishoptableb,rooktableb,queentableb,kingtableb]
    piecevalues = [100,320,330,500,900]


    def evaluate_board(board2):
        if board2.is_checkmate():
            if board2.turn:
                return 99999
            else:
                return -99999
        if board2.is_stalemate():
            return 0
        if board2.is_insufficient_material():
            return 0

        wp = len(board2.pieces(chess.PAWN, chess.WHITE))
        bp = len(board2.pieces(chess.PAWN, chess.BLACK))
        wn = len(board2.pieces(chess.KNIGHT, chess.WHITE))
        bn = len(board2.pieces(chess.KNIGHT, chess.BLACK))
        wb = len(board2.pieces(chess.BISHOP, chess.WHITE))
        bb = len(board2.pieces(chess.BISHOP, chess.BLACK))
        wr = len(board2.pieces(chess.ROOK, chess.WHITE))
        br = len(board2.pieces(chess.ROOK, chess.BLACK))
        wq = len(board2.pieces(chess.QUEEN, chess.WHITE))
        bq = len(board2.pieces(chess.QUEEN, chess.BLACK))
        wking = len(board2.pieces(chess.KING, chess.WHITE))
        bking = len(board2.pieces(chess.KING, chess.BLACK))

        material = 100 * (wp - bp) + 320 * (wn - bn) + 330 * (wb - bb) + 500 * (wr - br) + 900 * (wq - bq) + 20000*(wking-bking)
        pawnsq = sum([pawntablew[i] for i in board2.pieces(chess.PAWN, chess.WHITE)])
        pawnsq = pawnsq + sum([-pawntableb[i] for i in board2.pieces(chess.PAWN, chess.BLACK)])
        knightsq = sum([knightstablew[i] for i in board2.pieces(chess.KNIGHT, chess.WHITE)])
        knightsq = knightsq + sum([-knightstableb[i] for i in board2.pieces(chess.KNIGHT, chess.BLACK)])
        bishopsq = sum([bishoptablew[i] for i in board2.pieces(chess.BISHOP, chess.WHITE)])
        bishopsq = bishopsq + sum([-bishoptableb[i] for i in board2.pieces(chess.BISHOP, chess.BLACK)])
        rooksq = sum([rooktablew[i] for i in board2.pieces(chess.ROOK, chess.WHITE)])
        rooksq = rooksq + sum([-rooktableb[i] for i in board2.pieces(chess.ROOK, chess.BLACK)])
        queensq = sum([queentablew[i] for i in board2.pieces(chess.QUEEN, chess.WHITE)])
        queensq = queensq + sum([-queentableb[i] for i in board2.pieces(chess.QUEEN, chess.BLACK)])
        kingsq = sum([kingtablew[i] for i in board2.pieces(chess.KING, chess.WHITE)])
        kingsq = kingsq + sum([-kingtableb[i] for i in board2.pieces(chess.KING, chess.BLACK)])

        squarevalue =pawnsq+knightsq+bishopsq+rooksq+queensq+kingsq
        #oppmob=len(list(board2.legal_moves))

        #mob=len(list(board2.legal_moves))
        #mobility=-15*(oppmob)
        eval = material + squarevalue #+mobility

        if board2.turn:
            return eval
        else:
            return -eval


    def evaluatemove(board,move):
        movingpiece = board.piece_type_at(move.from_square)
        score=0
        capturedpiece = board.piece_type_at(move.to_square)


        if board2.turn:
            score = score - tablesw[movingpiece-1][move.from_square]
            score=score + tablesw[movingpiece-1][move.to_square]


        else:
            score = score - tablesb[movingpiece-1][move.from_square]
            score= score + tablesb[movingpiece-1][move.to_square]

        #mob=len(list(board.legal_moves))
        board.push(move)
        if board.is_checkmate():
            board.pop()
            if board.turn:
                return 99999
            else:
                return -99999
        if board.is_insufficient_material() or board.is_stalemate():
            board.pop()
            if board.turn:
                return stalemateconst
            else:
                return -stalemateconst
        #oppmob=len(list(board.legal_moves))
        board.pop()
        #mobilitybuff=5*(mob-oppmob)
        materialgain=0
        if capturedpiece != None:
            materialgain=piecevalues[capturedpiece-1]
        score=score+materialgain#+mobilitybuff
        return score

    def abmax(board, move, alpha, beta, maximize, depthleft):
        if depthleft == 0:
            score = evaluatemove(board, move)
            if maximize:
                score = -score
            return move, score
        pushed = False
        if move is not None:
            thismovescore = evaluatemove(board, move)
            board.push(move)
            pushed = True
        else:
            thismovescore = None

        if board.legal_moves.count() == 0:
            board.pop()
            if move is not None:
                if maximize:
                    thismovescore = -thismovescore
                return move, thismovescore
            else:
                return ("", 0)

        if maximize:
            maxScore = -10000
            for move in board.legal_moves:
                score = abmax(board, move, alpha, beta, False, depthleft - 1)[1]
                if (thismovescore is not None):
                    score -= thismovescore
                if score > maxScore:
                    bestmove = move
                maxScore = max(score, maxScore)
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
            if pushed:
                board.pop()
            return (bestmove, maxScore)
        else:
            minScore = 10000
            for move in board.legal_moves:
                score = abmax(board, move, alpha, beta, True, depthleft - 1)[1]
                if (thismovescore is not None):
                    score += thismovescore
                if score < minScore:
                    worstMove = move
                minScore = min(score, minScore)
                beta = min(beta, score)
                if beta <= alpha:
                    break
            if pushed:
                board.pop()
            return worstMove, minScore

    def selectmove(board, depth):
        alpha = -100000
        beta = 100000
        result = abmax(board,None,alpha, beta,True, depth)
        return result[0].uci()
    def stalematerate(board):
        wp = len(board.pieces(chess.PAWN, chess.WHITE))
        bp = len(board.pieces(chess.PAWN, chess.BLACK))
        wn = len(board.pieces(chess.KNIGHT, chess.WHITE))
        bn = len(board.pieces(chess.KNIGHT, chess.BLACK))
        wb = len(board.pieces(chess.BISHOP, chess.WHITE))
        bb = len(board.pieces(chess.BISHOP, chess.BLACK))
        wr = len(board.pieces(chess.ROOK, chess.WHITE))
        br = len(board.pieces(chess.ROOK, chess.BLACK))
        wq = len(board.pieces(chess.QUEEN, chess.WHITE))
        bq = len(board.pieces(chess.QUEEN, chess.BLACK))

        material = 10 * (wp - bp) + 32 * (wn - bn) + 33 * (wb - bb) + 50 * (wr - br) + 90 * (wq - bq)
        if material>0:
            return -material
        if material<=0:
            return material

    stalemateconst=stalematerate(board)

    move = selectmove(board2, 5)
    move=chess.Move.from_uci(move)
    return move


def engine(board):
    board2 = board
    moves = list(board2.legal_moves)
    for move in moves:
        if (board2.is_checkmate()):
            return move.uci()

        if (board2.gives_check(move)):
            return move.uci()
        if (board2.is_capture(move)):
            return move.uci()
    move = random.choice(list(board.legal_moves))
    return move.uci()



def StockFish(board):
    engine = chess.engine.SimpleEngine.popen_uci("C:\Users\joels\Desktop\ECE 559\Chess RL Project\stockfish_15_x64.exe")
    move = engine.play(board, chess.engine.Limit(time=0.05))
    move=move.move.uci()
    move=chess.Move.from_uci(move)
    return move


def human(uci,board):
    print(board)

    move=input("input move")
    move=chess.Move.from_uci(move)
    return move