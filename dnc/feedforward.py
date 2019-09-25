import torch
import torch.nn as nn

# Fully connected neural network with multiple hidden layer
class FFNN(nn.Module):
    def __init__(self, input_size, output_size, num_layers=1, bias=True):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size+1, bias=bias)
        self.sigm = nn.Sigmoid()
        self.layers = []
        for i in range(0, num_layers):
            if num_layers==1:
                break
            self.layers.append(nn.Linear(output_size+1, output_size+1, bias=bias))
        self.fc2 = nn.Linear(output_size+1, output_size, bias=bias)

    # The parameter chx is maintained, but it is ignored
    def forward(self, x, chx):
        out = self.fc1(x)
        out = self.sigm(out)
        for i in range(0, len(self.layers)):
            out = self.layers[i](out)
            out = self.sigm(out)
        out = self.fc2(out)
        return out, chx
