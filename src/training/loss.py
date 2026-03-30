import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """
    Symmetric Contrastive Loss (InfoNCE) used in CLIP and Two-Tower recommenders.
    It maximizes the cosine similarity between the true (user, item) pair
    and minimizes the similarity for all other (user, item) pairs in the batch
    (In-Batch Negatives) by treating them as negatives.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        # Learnable temperature parameter (initialized to 0.07 similar to CLIP)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        
    def forward(self, user_embeddings, item_embeddings):
        """
        user_embeddings: [B, D]
        item_embeddings: [B, D]
        Both inputs must be L2-normalized prior to this.
        """
        # Cosine similarity is just a dot product when vectors are L2-normalized
        logits = torch.matmul(user_embeddings, item_embeddings.T)
        
        # Scale the logits by the learned temperature
        logit_scale = self.logit_scale.exp()
        logits = logits * logit_scale
        
        # Ground truth: the diagonal represents positive pairs (user[i] <-> item[i])
        batch_size = user_embeddings.shape[0]
        labels = torch.arange(batch_size, device=user_embeddings.device)
        
        # Symmetric loss: User->Item and Item->User
        loss_u = F.cross_entropy(logits, labels)
        loss_i = F.cross_entropy(logits.T, labels)
        
        return (loss_u + loss_i) / 2.0
