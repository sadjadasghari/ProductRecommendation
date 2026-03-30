import torch
import torch.nn as nn
from src.models.two_tower import UserEncoder
import time
import os

# Fix for Apple Silicon (M-series) Mac quantization engines
torch.backends.quantized.engine = 'qnnpack'

def export_edge_model():
    print("🚀 Initializing User Tower for Edge Export...")
    
    # Instantiate the identical architecture trained in the pipeline
    user_model = UserEncoder(item_dim=128, hidden_dim=256, output_dim=128)
    user_model.eval()
    
    # In a real pipeline, we'd load the checkpoint:
    # state_dict = torch.load("two_tower_weights.pt")
    # But for demonstration, we export the initialized weights.
    
    # 1. Measure Base Model parameters and size
    base_params = sum(p.numel() for p in user_model.parameters())
    print(f"📦 Base Model Parameters: {base_params:,}")
    
    # Save unquantized for size comparison
    torch.save(user_model.state_dict(), "edge_user_model_fp32.pt")
    fp32_size = os.path.getsize("edge_user_model_fp32.pt") / (1024*1024)
    print(f"💾 Base FP32 Size: {fp32_size:.2f} MB")
    
    # 2. Apply Dynamic INT8 Quantization (Optimized for Mobile CPU/DSP)
    print("\n⚙️ Applying Post-Training Dynamic INT8 Quantization...")
    # We quantize the heavy nn.Linear and nn.GRU layers
    quantized_model = torch.ao.quantization.quantize_dynamic(
        user_model, 
        {nn.GRU, nn.Linear}, 
        dtype=torch.qint8
    )
    
    # Save quantized model
    torch.save(quantized_model.state_dict(), "edge_user_model_int8.pt")
    int8_size = os.path.getsize("edge_user_model_int8.pt") / (1024*1024)
    print(f"💾 Quantized INT8 Size: {int8_size:.2f} MB")
    print(f"🔥 Size Reduction: {(1 - int8_size/fp32_size)*100:.1f}%\n")
    
    # 3. Export to ONNX (for CoreML/XNNPACK cross-platform wrapping)
    print("🔄 Exporting to ONNX format...")
    # Dummy input sequence: Batch=1, SeqLen=15 (recent items), Dim=128
    dummy_input = torch.randn(1, 15, 128)
    
    # Exporting the FP32 model to ONNX (Quantization logic can be handled natively by ORT on device)
    torch.onnx.export(
        user_model, 
        dummy_input, 
        "edge_user_model.onnx", 
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['user_history_seq'],
        output_names=['user_embedding']
    )
    
    print("✅ Successfully exported `edge_user_model.onnx` ready for mobile wrappers.")

if __name__ == "__main__":
    export_edge_model()
