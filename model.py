import numpy as np
import torch
import torch.nn as nn


class VanillaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VanillaRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.Wxh = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.Whh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.Why = nn.Parameter(torch.randn(output_size, hidden_size) * 0.01)

        self.bh = nn.Parameter(torch.zeros(hidden_size, 1))
        self.by = nn.Parameter(torch.zeros(output_size, 1))

    def forward(self, inputs, h_prev=None):
        """
        Perform forward pass on the RNN

        :param inputs: List of input vectors, shape (input_shape, time_steps)
        :param h_prev: Initial hidden state, shape (hidden_size, 1)
        :return outputs: List of output vectors
                      h: Final hidden state
        """
        if h_prev is None:
            h_prev = torch.zeros(self.hidden_size, self.output_size)

        h = h_prev
        outputs = []

        for x in inputs:
            h = torch.tanh(torch.matmul(self.Wxh, x) + torch.matmul(self.Whh, h) + self.bh)
            y = torch.matmul(self.Why, h) + self.by
            outputs.append(y)
        return outputs, h


if __name__ == '__main__':
    input_size = 2
    hidden_size = 10
    output_size = 1
    time_steps = 4

    rnn = VanillaRNN(input_size, hidden_size, output_size)
    inputs = [torch.randn(input_size, 1) for _ in range(time_steps)]
    h_prev = torch.zeros(hidden_size, 1)
    outputs, h_final = rnn(inputs, h_prev)

    for t, y in enumerate(outputs):
        print(f'Time step {t}: {y.flatten()}')
    print('\nFinal hidden state:')
    print(h_final.flatten())
