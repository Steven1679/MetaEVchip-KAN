import torch
import torch.nn as nn
from kan import KAN


class InverseKAN(nn.Module):
    def __init__(self, input_size):
        super(InverseKAN, self).__init__()
        self.src_len = input_size  # input dimension
        self.tgt_len = 1  # output dimension
        self.hidden_len = 70
        self.kan = KAN([self.src_len,
                        self.hidden_len, self.hidden_len, self.hidden_len,
                        self.tgt_len])

    # forward propagation
    def forward(self, x):
        x = self.kan(x)
        x = torch.sigmoid(x)
        return x

