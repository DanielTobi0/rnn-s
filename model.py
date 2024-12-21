import torch
import torch.nn as nn
from utils import device


class TabularVanillaRNN(nn.Module):
    """
    simple vanilla rnn.

    self.Wxh : input-to-hidden
    self.Whh: hidden-to-hidden
    self.Why: hidden-to-output
    self.bh: hidden bias
    self.by: output bias
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(TabularVanillaRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.Wxh = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.Whh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.Why = nn.Parameter(torch.randn(output_size, hidden_size) * 0.01)

        # bias
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
        batch_size = inputs.size(0)

        if h_prev is None:
            h_prev = torch.zeros(batch_size, self.hidden_size).to(inputs.device)

        h = h_prev

        h = torch.tanh(torch.matmul(inputs, self.Wxh.t()) + torch.matmul(h, self.Whh.t()) + self.bh.t())
        y = torch.matmul(h, self.Why.t() + self.by.t())
        return y


class TabularVanilla2(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, batch_size):
        super(TabularVanilla2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.output_size = output_size

        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


class TabularGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, batch_size, dropout, bidirectional=True):
        super(TabularGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.output_size = output_size
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=True
        )

        self.num_directions = 2 if self.bidirectional else 1
        self.fc = nn.Linear(hidden_size * self.num_directions, self.output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        h0 = torch.zeros(
            self.num_layers * self.num_directions,
            x.size(0),
            self.hidden_size
        ).to(device)

        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out


class TabularLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, batch_size, dropout, bidirectional=True):
        super(TabularLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.output_size = output_size
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
            bidirectional=True
        )

        self.num_directions = 2 if self.bidirectional else 1
        self.fc = nn.Linear(hidden_size * self.num_directions, self.output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        h0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * self.num_directions, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out