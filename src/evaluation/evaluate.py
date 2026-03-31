import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.two_tower import UserEncoder, ItemEncoder
from src.data.multimodal_dataset import MultimodalRetailDataset, collate_multimodal_batch
import time
import math
import os

# Fix for Apple Silicon (M-series) Mac quantization engines
torch.backends.quantized.engine = 'qnnpack'

def calculate_metrics(user_embs, item_embs, k=10):
    """
    Computes Hit Rate at K (HR@K) and NDCG@K.
    user_embs: [N, D]
    item_embs: [N, D]
    The ground truth item for user i is item i.
    """
    N = user_embs.size(0)
    
    # Compute cosine similarity matrix: [N, N]
    # Assuming embeddings are already L2 normalized
    sim_matrix = torch.matmul(user_embs, item_embs.T)
    
    hits = 0
    ndcg = 0.0
    
    for i in range(N):
        # Get similarities for user i
        user_sims = sim_matrix[i]
        
        # Sort indices descending
        ranked_indices = torch.argsort(user_sims, descending=True)
        
        # Ground truth index is i
        # Find the rank of the ground truth item (0-indexed)
        rank = (ranked_indices == i).nonzero(as_tuple=True)[0].item()
        
        # If it's within top K
        if rank < k:
            hits += 1
            ndcg += 1.0 / math.log2(rank + 2) # rank is 0-indexed, so rank+2 for log2(rank+1 where rank is 1-indexed)
            
    hr_at_k = hits / N
    ndcg_at_k = ndcg / N
    
    return hr_at_k, ndcg_at_k

def run_evaluation():
    print("🚀 Initializing Evaluation Pipeline...")
    
    # 1. Load Item Encoder (Using Random Init since we are running functional validation)
    print("Loading Item Tower (uninitialized weights for functional simulation)...")
    item_tower = ItemEncoder(fused_dim=128)
    item_tower.eval()
    
    # 2. Load FP32 User Encoder
    print("Loading FP32 User Tower...")
    user_model_fp32 = UserEncoder(item_dim=128, hidden_dim=256, output_dim=128)
    if os.path.exists("edge_user_model_fp32.pt"):
        user_model_fp32.load_state_dict(torch.load("edge_user_model_fp32.pt", weights_only=False))
    user_model_fp32.eval()
    
    # 3. Load INT8 User Encoder
    print("Loading INT8 User Tower...")
    user_model_int8 = UserEncoder(item_dim=128, hidden_dim=256, output_dim=128)
    user_model_int8 = torch.ao.quantization.quantize_dynamic(
        user_model_int8, 
        {nn.Linear}, 
        dtype=torch.qint8
    )
    if os.path.exists("edge_user_model_int8.pt"):
        user_model_int8.load_state_dict(torch.load("edge_user_model_int8.pt", weights_only=False))
    user_model_int8.eval()
    
    # 4. Generate Test Data
    print("\nGenerating Test Set...")
    test_ds = MultimodalRetailDataset(num_samples=500)
    test_loader = DataLoader(
        test_ds, 
        batch_size=100, 
        shuffle=False, 
        collate_fn=collate_multimodal_batch
    )
    
    # 5. Extract Embeddings
    all_target_item_embs = []
    
    all_user_embs_fp32 = []
    all_user_embs_int8 = []
    
    fp32_latency = 0.0
    int8_latency = 0.0
    
    print("Extracting Embeddings...")
    with torch.no_grad():
        for batch in test_loader:
            history, target_img, target_txt_ids, target_txt_mask = batch
            
            # Extract item embedding
            item_emb = item_tower(target_img, target_txt_ids, target_txt_mask)
            all_target_item_embs.append(item_emb)
            
            # Predict user embeddings (FP32)
            start_t = time.time()
            user_emb_fp32 = user_model_fp32(history)
            fp32_latency += (time.time() - start_t)
            all_user_embs_fp32.append(user_emb_fp32)
            
            # Predict user embeddings (INT8)
            start_t = time.time()
            user_emb_int8 = user_model_int8(history)
            int8_latency += (time.time() - start_t)
            all_user_embs_int8.append(user_emb_int8)
            
    # Concat
    all_target_item_embs = torch.cat(all_target_item_embs, dim=0)
    all_user_embs_fp32 = torch.cat(all_user_embs_fp32, dim=0)
    all_user_embs_int8 = torch.cat(all_user_embs_int8, dim=0)
    
    # 6. Evaluate
    print("\n📊 Computing Metrics (HR@10, NDCG@10)...")
    
    hr_fp32, ndcg_fp32 = calculate_metrics(all_user_embs_fp32, all_target_item_embs, k=10)
    hr_int8, ndcg_int8 = calculate_metrics(all_user_embs_int8, all_target_item_embs, k=10)
    
    print("\n================ EVALUATION RESULTS ================")
    print(f"[FP32 Model]: HR@10: {hr_fp32:.4f} | NDCG@10: {ndcg_fp32:.4f}")
    print(f"   Average Latency per batch: {(fp32_latency/len(test_loader))*1000:.2f} ms")
    
    print(f"\n[INT8 Model]: HR@10: {hr_int8:.4f} | NDCG@10: {ndcg_int8:.4f}")
    print(f"   Average Latency per batch: {(int8_latency/len(test_loader))*1000:.2f} ms")
    print("====================================================\n")
    
if __name__ == "__main__":
    run_evaluation()
