# D:\DecoderKAN\src\KAN_TP_Transformer.py
class HyperParams:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.filter = 512
        self.n_layers = 6
        self.n_heads = 8
        self.hidden = 512
        self.dropout = 0.1
# Remove: from utils.lib import HyperParams