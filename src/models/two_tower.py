import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel
import math

class ItemEncoder(nn.Module):
    """
    Multimodal Item Tower.
    Fuses vision features (MobileNetV3) and text features (HuggingFace AutoModel)
    into a single dense item representation. 
    """
    def __init__(self, text_model_name="sentence-transformers/all-MiniLM-L6-v2", fused_dim=128):
        super().__init__()
        
        # Vision backbone (Vision Transformer)
        # Using vit_b_16 (weights: DEFAULT) replacing MobileNetV3
        self.vision_extractor = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        # We strip the classification head so it outputs raw 768-D ViT embeddings
        self.vision_extractor.heads = nn.Identity()
        vision_out_dim = 768  # ViT-B/16 hidden size
        
        # Text backbone (Transformer)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_out_dim = self.text_encoder.config.hidden_size # usually 384 for MiniLM
        
        # Multimodal Fusion layer
        self.fusion_mlp = nn.Sequential(
            nn.Linear(vision_out_dim + text_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, fused_dim)
        )
        
    def forward(self, images, text_input_ids, text_attn_mask):
        # 1. Vision Features (ViT automatically outputs 768-D flat vector)
        # images: [B, C, 224, 224]
        v_features = self.vision_extractor(images)
        
        # 2. Text Features
        transformer_out = self.text_encoder(input_ids=text_input_ids, attention_mask=text_attn_mask)
        # Apply Mean Pooling over sequence, leveraging attention mask
        token_embeddings = transformer_out[0]
        input_mask_expanded = text_attn_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        t_features = sum_embeddings / sum_mask # [B, 384]
        
        # 3. Fusion
        fused = torch.cat([v_features, t_features], dim=1)
        item_emb = self.fusion_mlp(fused)
        
        # L2 Normalize for cosine similarity lookup
        item_emb = nn.functional.normalize(item_emb, p=2, dim=1)
        return item_emb


class PositionalEncoding(nn.Module):
    """Injects positional information into sequence embeddings for the Transformer"""
    def __init__(self, d_model: int, max_len: int = 50):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]
        return x

class CustomTransformerEncoderLayer(nn.Module):
    """A clean, PyTorch PTQ-safe Transformer layer bypassing native C++ fastpaths"""
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.GELU()

    def forward(self, src):
        # Attention
        src2, _ = self.self_attn(src, src, src)
        src = src + src2
        src = self.norm1(src)
        # FFN
        src2 = self.linear2(self.activation(self.linear1(src)))
        src = src + src2
        src = self.norm2(src)
        return src

class UserEncoder(nn.Module):
    """
    Sequential User Tower (Self-Attentive).
    Takes a sequence of historical item embeddings the user interacted with,
    processing them via Transformer Encoder blocks.
    """
    def __init__(self, item_dim=128, hidden_dim=256, output_dim=128, max_seq_len=20, context_dim=16):
        super().__init__()
        
        self.pos_encoder = PositionalEncoding(item_dim, max_seq_len)
        
        # 2 layers is lightweight enough for mobile edge execution
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(item_dim, nhead=4, dim_feedforward=hidden_dim) 
            for _ in range(2)
        ])
        
        self.projection = nn.Sequential(
            nn.Linear(item_dim + context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, history_item_embeddings, context_features):
        # history_item_embeddings: [B, seq_len, item_dim]
        # context_features: [B, context_dim]
        
        # 1. Add Positional Encodings
        x = self.pos_encoder(history_item_embeddings)
        
        # 2. Self-Attention Processing
        for layer in self.layers:
            x = layer(x)
            
        # 3. Inject Feature Store Real-Time Context & Project
        user_intent = torch.cat([x[:, -1, :], context_features], dim=1)
        user_emb = self.projection(user_intent)
        
        # 4. L2 Normalize
        user_emb = nn.functional.normalize(user_emb, p=2, dim=1)
        return user_emb


class TwoTowerRecSys(nn.Module):
    """
    Wrapper for training the two towers jointly using contrastive learning.
    """
    def __init__(self, embed_dim=128):
        super().__init__()
        self.item_tower = ItemEncoder(fused_dim=embed_dim)
        self.user_tower = UserEncoder(item_dim=embed_dim, output_dim=embed_dim)
        
    def forward(self, user_history_embeddings, target_images, target_text_ids, target_text_mask, context_features):
        user_emb = self.user_tower(user_history_embeddings, context_features)
        target_item_emb = self.item_tower(target_images, target_text_ids, target_text_mask)
        return user_emb, target_item_emb
