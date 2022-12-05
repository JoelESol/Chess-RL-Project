import csv
import io
import json
import chess
import chess.pgn
openings={}
with open("opening_books/a.tsv") as file:
    tsv_file = csv.reader(file, delimiter="\t")

    for line in tsv_file:
        name=line[0]
        print(name)
        moves=io.StringIO(line[-1])
        game=chess.pgn.read_game(moves)
        board=game.board()
        for move in game.mainline_moves():
            board.push(move)
        fen=board.fen()
        openings[name]=fen

with open("opening_books/b.tsv") as file:
    tsv_file = csv.reader(file, delimiter="\t")

    for line in tsv_file:
        name=line[0]
        print(name)
        moves=io.StringIO(line[-1])
        game=chess.pgn.read_game(moves)
        board=game.board()
        for move in game.mainline_moves():
            board.push(move)
        fen=board.fen()
        openings[name]=fen

with open("opening_books/c.tsv") as file:
    tsv_file = csv.reader(file, delimiter="\t")

    for line in tsv_file:
        name=line[0]
        print(name)
        moves=io.StringIO(line[-1])
        game=chess.pgn.read_game(moves)
        board=game.board()
        for move in game.mainline_moves():
            board.push(move)
        fen=board.fen()
        openings[name]=fen

with open("opening_books/d.tsv") as file:
    tsv_file = csv.reader(file, delimiter="\t")

    for line in tsv_file:
        name=line[0]
        print(name)
        moves=io.StringIO(line[-1])
        game=chess.pgn.read_game(moves)
        board=game.board()
        for move in game.mainline_moves():
            board.push(move)
        fen=board.fen()
        openings[name]=fen

with open("opening_books/e.tsv") as file:
    tsv_file = csv.reader(file, delimiter="\t")

    for line in tsv_file:
        name=line[0]
        print(name)
        moves=io.StringIO(line[-1])
        game=chess.pgn.read_game(moves)
        board=game.board()
        for move in game.mainline_moves():
            board.push(move)
        fen=board.fen()
        openings[name]=fen

with open("opening_books/opening_fens.json", 'w') as outfile:
    json.dump(openings, outfile, indent=4)