import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import AutoTokenizer

class MultimodalRetailDataset(Dataset):
    """
    Simulated large-scale multimodal retail dataset.
    Returns:
    - User History (Sequence of Item Embeddings)
    - Target Positive Item (Image tensor + Text features)
    """
    def __init__(self, num_samples=10000, seq_len=10, max_text_len=32, embed_dim=128, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.max_text_len = max_text_len
        self.embed_dim = embed_dim
        
        # We load a real tokenizer to accurately simulate inputs to the transformer text backbone
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Sample phrases to generate some dynamic data
        self.sample_descriptions = [
            "A comfortable cotton graphic t-shirt in bright red.",
            "Mid-century modern leather sofa with wooden legs style.",
            "Waterproof hiking boots perfect for outdoor trail walks.",
            "Stainless steel smart watch with heart rate monitor.",
            "Minimalist ceramic coffee mug for morning espresso."
        ]
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # 1. User History (Simulate last N item embeddings watched by user)
        history_embeddings = torch.randn(self.seq_len, self.embed_dim)
        
        # 2. Target Positive Item (The next item they clicked)
        target_image = torch.randn(3, 224, 224)
        
        # 3. Text features
        desc = self.sample_descriptions[idx % len(self.sample_descriptions)]
        tokenized = self.tokenizer(
            desc,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt"
        )
        
        
        # 4. Feature Store Real-time Context (Appends location, time, device metadata)
        context_features = torch.randn(16)
        
        return {
            "history_embs": history_embeddings,
            "target_img": target_image,
            "target_txt_ids": tokenized['input_ids'].squeeze(0),
            "target_txt_mask": tokenized['attention_mask'].squeeze(0),
            "context_features": context_features
        }

# Collate function for DataLoader
def collate_multimodal_batch(batch):
    histories = torch.stack([b["history_embs"] for b in batch])
    images = torch.stack([b["target_img"] for b in batch])
    text_ids = torch.stack([b["target_txt_ids"] for b in batch])
    text_masks = torch.stack([b["target_txt_mask"] for b in batch])
    contexts = torch.stack([b["context_features"] for b in batch])
    
    return histories, images, text_ids, text_masks, contexts
