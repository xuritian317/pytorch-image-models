import torch
import torch.nn as nn
import torch.nn.functional as F


class ConLossEntropy(nn.Module):
    def __init__(self):
        super(ConLossEntropy, self).__init__()

    def forward(self, features, labels):
        loss = self.con_loss(features, labels)
        return loss

    def con_loss(self, features: torch.Tensor, labels: torch.Tensor)-> torch.Tensor:
        B, _ = features.shape
        features = F.normalize(features)
        cos_matrix = features.mm(features.t())
        pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
        neg_label_matrix = 1 - pos_label_matrix
        pos_cos_matrix = 1 - cos_matrix
        neg_cos_matrix = cos_matrix - 0.4
        # neg_cos_matrix[neg_cos_matrix < 0] = 0
        neg_cos_matrix = neg_cos_matrix.clamp(min=0.0)
        loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
        loss /= (B * B)
        return loss.mean()
