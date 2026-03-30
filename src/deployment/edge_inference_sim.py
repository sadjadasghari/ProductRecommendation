import torch
import torch.nn as nn
from src.models.two_tower import UserEncoder
import numpy as np
import time

# Fix for Apple Silicon (M-series) Mac quantization engines
torch.backends.quantized.engine = 'qnnpack'

def simulate_edge_latency():
    print("📱 Starting On-Device Inference Simulator (iOS/Android Mocks)...\n")
    
    # 1. Load the Quantized Edge Model
    user_model = UserEncoder(item_dim=128, hidden_dim=256, output_dim=128)
    edge_model = torch.ao.quantization.quantize_dynamic(
        user_model, {nn.GRU, nn.Linear}, dtype=torch.qint8
    )
    edge_model.eval()
    
    # 2. Simulate Local Vector DB (e.g. FAISS running on device storage)
    print("📦 Building simulated local Item Cache (100k items) for vector lookup...")
    num_items = 100000
    catalog_dim = 128
    # We use numpy to simulate FAISS fast exact search
    cached_catalog = np.random.randn(num_items, catalog_dim).astype(np.float32)
    # Normalize for cosine similarity (dot product)
    cached_catalog /= np.linalg.norm(cached_catalog, axis=1, keepdims=True)
    
    print("\n⚡ Running Local Live Profiling:")
    latencies = []
    
    for i in range(100):
        # The user looks at 12 items securely on phone. No data sent to cloud.
        user_history = torch.randn(1, 12, 128)
        
        t0 = time.perf_counter()
        
        # Step A: Generate User Vector
        with torch.no_grad():
            user_embedding = edge_model(user_history)
            
        # Step B: Nearest Neighbor Search on local catalog
        query_vec = user_embedding.numpy()
        # Cosine similarity scores
        similarities = np.dot(cached_catalog, query_vec.T).flatten()
        
        # Get top-5 recs computationally
        top_k = 5
        # argpartition is O(N) compared to argsort O(N log N), making fast mobile search realistic
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])][::-1]
        
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000) # milliseconds
        
    avg_latency = np.mean(latencies[10:]) # exclude warmup
    p95_latency = np.percentile(latencies[10:], 95)
    
    print(f"📊 Benchmarks over 100 interaction cycles:")
    print(f"   ► Average End-to-End Latency: {avg_latency:.2f} ms")
    print(f"   ► P95 Latency: {p95_latency:.2f} ms")
    
    if p95_latency < 50.0:
        print("\n✅ Edge Latency Goal (<50ms) achieved! User experience is real-time locally.")
    else:
        print("\n⚠️ Warning: Latency above 50ms budget, further ONNX runtime tuning required.")

if __name__ == "__main__":
    simulate_edge_latency()
