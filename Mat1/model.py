import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Nonlinearity(nn.Module):
    def __init__(self):
        super(Nonlinearity, self).__init__()

    def forward(self, x):
        return F.relu(x)
        #return F.tanh(x)

class Net(nn.Module):
    def __init__(self, dim, num_classes=2):
        super(Net, self).__init__()
        bias = False
        k = 1024
        self.dim = dim
        self.width = k

        self.features = nn.Sequential(
            nn.Linear(dim, k, bias=bias),
            Nonlinearity(),
            #nn.Linear(k, k, bias=bias),
            #Nonlinearity(),
        )

        self.classifier = nn.Sequential(           
            nn.Linear(k, num_classes, bias=bias)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            #nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            nn.init.xavier_uniform_(m.weight)
            if m.bias:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x