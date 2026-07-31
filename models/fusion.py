import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    def __init__(self, in_dims, hidden_dim, num_classes):
        super(ConcatFusion, self).__init__()
        total_dim = sum(in_dims)
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, features):
        x = torch.cat(features, dim=-1)
        logits = self.mlp(x)
        return logits
