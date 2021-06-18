import numpy as np
import yaml
from . import tfprocess
from collections import OrderedDict


class KerasNet:

    def __init__(self, model_file="Agent/128x10-t60-2-5300.pb.gz", cfg_file="Agent/configs/example.yaml"):

        with open(cfg_file, "rb") as f:
            cfg = f.read()

        cfg = yaml.safe_load(cfg)

        tfp = tfprocess.TFProcess(cfg, gpu=True)
        tfp.init_net_v2()
        tfp.replace_weights_v2(model_file)

        self.model = tfp.model 

    def _softmax(self, x, softmax_temp=1.0):
        e_x = np.exp((x - np.max(x))/softmax_temp)
        return e_x / e_x.sum(axis=0)

    def get_policy(self, leela_board):
        
        input_planes = leela_board.lcz_features()
        model_input = input_planes.reshape(1, 112, 64)
        model_output = self.model.predict(model_input)
        policy = model_output[0][0]

        return self._softmax(policy)


    def _evaluate(self, leela_board):

        input_planes = leela_board.lcz_features()
        model_input = input_planes.reshape(1, 112, 64)
        model_output = self.model.predict(model_input)

        policy = model_output[0][0]

        legal_uci = [m.uci() for m in leela_board.generate_legal_moves()]
        
        if legal_uci:
            legal_indexes = leela_board.lcz_uci_to_idx(legal_uci)
            softmaxed = self._softmax(policy[legal_indexes])
            softmaxed_aspython = map(float, softmaxed)
            policy_legal = OrderedDict(sorted(zip(legal_uci, softmaxed_aspython),
                                        key = lambda mp: (mp[1], mp[0]),
                                        reverse=True))

        else:
            policy_legal = OrderedDict()

        return policy_legal

    def get_move(self, board):

        move_dict = self._evaluate(board)
    
        for k, v in move_dict.items():
            move = k
            break

        return move 
