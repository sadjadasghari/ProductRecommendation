import torch
from torch.utils.data import Dataset
import numpy as np

class MultimodalRetailDataset(Dataset):
    """
    Simulated large-scale multimodal retail dataset.
    Returns:
    - User History (Sequence of Item Embeddings)
    - Target Positive Item (Image tensor + Text tokens)
    """
    def __init__(self, num_samples=10000, seq_len=10, vocab_size=30000, max_text_len=20, embed_dim=128):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.max_text_len = max_text_len
        self.embed_dim = embed_dim
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Simulate user history: a sequence of previous item embeddings the user clicked
        # Note: In a real pipeline, we'd lookup these embeddings dynamically or from a precomputed store.
        history_embeddings = torch.randn(self.seq_len, self.embed_dim)
        
        # Simulate Target Positive Item (The next item they actually clicked)
        # 1. Image (3 channels, 224x224 for MobileNetV3)
        target_image = torch.randn(3, 224, 224)
        
        # 2. Text Description Tokens
        target_text_tokens = torch.randint(0, self.vocab_size, (self.max_text_len,))
        text_length = torch.tensor(self.max_text_len, dtype=torch.long)
        
        return {
            "history_embs": history_embeddings,
            "target_img": target_image,
            "target_txt": target_text_tokens,
            "target_len": text_length
        }

# Collate function for DataLoader
def collate_multimodal_batch(batch):
    histories = torch.stack([b["history_embs"] for b in batch])
    images = torch.stack([b["target_img"] for b in batch])
    texts = torch.stack([b["target_txt"] for b in batch])
    lengths = torch.stack([b["target_len"] for b in batch])
    
    return histories, images, texts, lengths
