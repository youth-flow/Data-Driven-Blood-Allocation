"""Flat v7: two independently MSE-supervised small MLPs."""

from torch import nn


class FlatMSE(nn.Module):
    def __init__(self, patient_hidden, residual_hidden, dropout=0.0):
        super().__init__()
        self.patient = self._mlp(3, patient_hidden, 1, dropout)
        self.residual = self._mlp(7, residual_hidden, 1, dropout)

    @staticmethod
    def _mlp(input_dim, hidden, output_dim, dropout):
        layers = []
        current = input_dim
        for width in hidden:
            layers.extend([nn.Linear(current, int(width)), nn.ReLU(), nn.Dropout(float(dropout))])
            current = int(width)
        layers.append(nn.Linear(current, output_dim))
        return nn.Sequential(*layers)

    def patient_prediction(self, patient):
        return self.patient(patient).squeeze(-1)

    def residual_prediction(self, context):
        return self.residual(context).squeeze(-1)
