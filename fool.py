from lcztools import load_network, LeelaBoard
from . import kerasNet
from . import policy_index 


net = kerasNet.KerasNet()
board = LeelaBoard()


while  board.is_game_over() == False:

    print(board)
    move_dict = net._evaluate(board)
    for k, v in move_dict[0].items():
        move = k
        break
    board.push_uci(move)

    
print(board)
