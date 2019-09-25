import torch
import torch.nn as nn
import collections

# Fully connected neural network with multiple hidden layer
class FFNN(nn.Module):
    def __init__(self, input_size, output_size, num_layers=1, bias=True):
        super(FFNN, self).__init__()
        self.layers = collections.OrderedDict()
        self.layers["layer_input"] = nn.Linear(input_size, output_size+1, bias=bias)
        self.layers["sigmoid"] = nn.Sigmoid()
        for i in range(0, num_layers):
            if num_layers==1:
                break
            self.layers["hidden_{}".format(i+1)] = nn.Linear(output_size+1, output_size+1, bias=bias)
            self.layers["sigmoid_{}".format(i+1)] = nn.Sigmoid()
        self.layers["layer_output"] = nn.Linear(output_size+1, output_size, bias=bias)
        self.model = nn.Sequential(self.layers)

    # The parameter chx is maintained, but it is ignored
    def forward(self, x, chx):
        out = self.model.forward(x)
        return out, chx
