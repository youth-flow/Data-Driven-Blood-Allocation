"""Fixed patient-level seven-class PMF network for Flat-ETO."""

from torch import nn


class FlatPatientPMF(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        current = int(input_dim)
        for width in hidden_dims:
            layers.extend([nn.Linear(current, int(width)), nn.ReLU(), nn.Dropout(float(dropout))])
            current = int(width)
        layers.append(nn.Linear(current, 7))
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        return self.network(features)
