import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np


class LSTMForecast(nn.Module):
    def __init__(self, hidden_dim):
        super(LSTMForecast, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, num_layers=1)
        self.fc_mu = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=32, out_features=1),
        )

        self.fc_sigma = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=32, out_features=1),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(1, x.size(0), self.hidden_dim).requires_grad_()

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out


def testy(df):
    new_df = df * (1 + np.random.normal(loc=0, scale=0.4)) + np.random.uniform(low=-10**4, high=10**4)
    return new_df
