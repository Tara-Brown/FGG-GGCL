import torch.nn as nn
import torch
import logging
logger = logging.getLogger()
import torch.nn.functional as F

class NCESoftmaxLoss(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, similarity):
        batch_size = similarity.size(0) // 2
        label = torch.tensor([(batch_size + i) % (batch_size*2) for i in range(batch_size*2)]).cuda().long()
        loss = self.criterion(similarity, label)
        return loss


class FlatNCE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, similarity):
        pass


class MultiPosConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(MultiPosConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        :param features: Tensor of shape [B, D]
        :param labels: Tensor of shape [B]
        :return: Scalar contrastive loss
        """
        device = features.device
        features = F.normalize(features, dim=1)
        batch_size = features.shape[0]

        # Compute similarity matrix
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # Mask self-similarity
        logits_mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(~logits_mask, float('-inf'))

        # Build positive mask
        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float().to(device)
        pos_mask = pos_mask * logits_mask.float()

        # Normalize positive counts
        pos_counts = pos_mask.sum(1).clamp(min=1)
        log_prob = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_counts

        loss = -mean_log_prob_pos.mean()
        return loss
