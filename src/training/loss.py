import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """
    Symmetric Contrastive Loss with Hard Negative Mining (Additive Margin).
    By subtracting a margin from the true positive similarities, we force the model 
    to separate positive pairs from the hardest in-batch negative pairs more aggressively.
    """
    def __init__(self, temperature=0.07, margin=0.15):
        super().__init__()
        self.margin = margin
        # Learnable temperature parameter (initialized to 0.07 similar to CLIP)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        
    def forward(self, user_embeddings, item_embeddings):
        """
        user_embeddings: [B, D]
        item_embeddings: [B, D]
        Both inputs must be L2-normalized prior to this.
        """
        # Cosine similarity matrix (since vectors are L2-normalized)
        cosine_sim = torch.matmul(user_embeddings, item_embeddings.T)
        
        batch_size = user_embeddings.shape[0]
        labels = torch.arange(batch_size, device=user_embeddings.device)
        
        # Additive Margin for Hard Negative Separation (CosFace mechanics)
        # Subtract the margin directly from the true positive logits
        cosine_sim[labels, labels] -= self.margin
        
        # Scale the logits by the learned temperature
        logit_scale = self.logit_scale.exp()
        logits = cosine_sim * logit_scale
        batch_size = user_embeddings.shape[0]
        labels = torch.arange(batch_size, device=user_embeddings.device)
        
        # Symmetric loss: User->Item and Item->User
        loss_u = F.cross_entropy(logits, labels)
        loss_i = F.cross_entropy(logits.T, labels)
        
        return (loss_u + loss_i) / 2.0
