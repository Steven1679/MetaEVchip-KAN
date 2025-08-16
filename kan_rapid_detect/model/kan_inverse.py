import torch.nn as nn
from kan import KAN


class InverseKAN(nn.Module):
    def __init__(self, args):
        super(InverseKAN, self).__init__()
        self.args = args
        self.src_len = 15  # input dimension
        self.tgt_len = 1  # output dimension
        self.hidden_len = 70
        self.kan = KAN([self.src_len,
                        self.hidden_len, self.hidden_len, self.hidden_len,
                        self.hidden_len, self.hidden_len, self.hidden_len,
                        self.tgt_len])

    # forward propagation
    def forward(self, x):
        x = self.kan(x)
        return x

