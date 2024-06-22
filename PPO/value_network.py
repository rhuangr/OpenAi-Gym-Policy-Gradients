import torch
import torch.nn as nn
from torch.optim import Adam


class value_network(nn.Module):
    def __init__(self,input_size, output_size, middle_size = 64, activation = nn.ReLU) -> None:
        super(value_network, self).__init__()

        self.layer1 = nn.Linear(input_size, middle_size)
        self.layer2 = nn.Linear(middle_size, middle_size )
        self.layer3 = nn.Linear(middle_size, output_size)
        self.activation = activation

    def forward(self, input):
        input = torch.as_tensor(input, dtype=torch.float32)
        x = self.layer1(input)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        return x