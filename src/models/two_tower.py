import torch
import torch.nn as nn
import torchvision.models as models

class ItemEncoder(nn.Module):
    """
    Multimodal Item Tower.
    Fuses vision features (MobileNetV3) and text features (simple Embedding/LSTM)
    into a single dense item representation. Designed to be run offline to build
    the local edge cache.
    """
    def __init__(self, vocab_size=30000, text_embed_dim=128, fused_dim=128):
        super().__init__()
        
        # Vision backbone (lightweight)
        # Using mobilenet_v3_small for fast extraction if run on edge, 
        # but typically this tower runs in the cloud.
        vision_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        # Remove the classification head
        self.vision_extractor = nn.Sequential(*list(vision_model.children())[:-1])
        vision_out_dim = 576  # MobileNetV3 small features
        
        # Text backbone
        self.text_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=text_embed_dim)
        self.text_encoder = nn.GRU(input_size=text_embed_dim, hidden_size=text_embed_dim, batch_first=True)
        
        # Multimodal Fusion layer
        self.fusion_mlp = nn.Sequential(
            nn.Linear(vision_out_dim + text_embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, fused_dim)
        )
        
    def forward(self, images, text_tokens, text_lengths):
        # 1. Vision Features
        # images: [B, C, H, W]
        v_features = self.vision_extractor(images)
        v_features = v_features.mean([2, 3])  # Global average pooling -> [B, 576]
        
        # 2. Text Features
        # text_tokens: [B, seq_len]
        t_embeds = self.text_embedding(text_tokens)     # [B, seq_len, embed_dim]
        # Pack sequence could be used here, but we simplify for export
        _, t_hidden = self.text_encoder(t_embeds)       # [1, B, embed_dim]
        t_features = t_hidden.squeeze(0)                # [B, embed_dim]
        
        # 3. Fusion
        fused = torch.cat([v_features, t_features], dim=1)
        item_emb = self.fusion_mlp(fused)
        
        # L2 Normalize for cosine similarity lookup
        item_emb = nn.functional.normalize(item_emb, p=2, dim=1)
        return item_emb


class UserEncoder(nn.Module):
    """
    Sequential User Tower.
    Takes a sequence of historical item embeddings the user interacted with,
    and predicts the next intended item embedding.
    DESIGNED FOR ON-DEVICE EDGE INFERENCE (CoreML/TFLite/ONNX).
    """
    def __init__(self, item_dim=128, hidden_dim=256, output_dim=128):
        super().__init__()
        
        # Must be lightweight for mobile deployment
        self.sequence_encoder = nn.GRU(
            input_size=item_dim,
            hidden_size=hidden_dim, 
            num_layers=1, 
            batch_first=True
        )
        
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, history_item_embeddings):
        # history_item_embeddings: [B, seq_len, item_dim]
        _, hidden = self.sequence_encoder(history_item_embeddings)
        user_emb = self.projection(hidden.squeeze(0))
        
        # L2 Normalize
        user_emb = nn.functional.normalize(user_emb, p=2, dim=1)
        return user_emb


class TwoTowerRecSys(nn.Module):
    """
    Wrapper for training the two towers jointly using contrastive learning.
    """
    def __init__(self, vocab_size=30000, embed_dim=128):
        super().__init__()
        self.item_tower = ItemEncoder(vocab_size=vocab_size, fused_dim=embed_dim)
        self.user_tower = UserEncoder(item_dim=embed_dim, output_dim=embed_dim)
        
    def forward(self, user_history_embeddings, target_images, target_text, target_len):
        """
        During training, we encode the user history and the target positive item.
        """
        user_emb = self.user_tower(user_history_embeddings)
        target_item_emb = self.item_tower(target_images, target_text, target_len)
        return user_emb, target_item_emb
