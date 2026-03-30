# Multimodal Edge RecSys: Implementation Plan

## Problem Statement
The goal is to build a large-scale **multimodal product recommendation system** (text descriptions + product images) that generates highly personalized item recommendations, but carefully engineered for **on-device deployment**. This requires bridging the gap between heavy, high-accuracy representation learning (like CLIP/SigLIP) and extremely constrained mobile edge environments (iOS CoreML / Android TFLite / ExecuTorch).

## User Review Required
> [!IMPORTANT]
> **Key Architecture Decisions:**
> 1. **Model Backbone:** I propose using a lightweight **Two-Tower Architecture**, utilizing a distilled MobileCLIP or tiny-ViT combined with a fast text encoder for the Item Tower, and a sequential GRU/Transformer for the User Behavior Tower.
> 2. **Edge Inference Strategy:** Rather than running the entire model on-device, we will run the **User Tower** dynamically on the edge to generate a "user embedding" based on real-time interactions, and retrieve nearest neighbors from a cached local vector database (e.g., SQLite-VSS or local FAISS) containing pre-computed **Item Embeddings**.
> 3. **Quantization:** We will apply Post-Training Quantization (PTQ) to INT8 to shrink the model size to < 10MB for mobile deployment.
> 
> *Please confirm if this Edge User-Tower + Cached Item-Tower approach aligns with your deployment constraints!*

## Proposed Changes

We will work exclusively within the `~/Workspace/ProductRecommendation` directory.

### 1. Data Pipeline & Representation Learning
- Define the multimodal dataset loader (Images + Metadata).
- Implement the Item Encoder (Image + Text Fusion).
- Implement the User Encoder (Sequential sequence of interacted items).
#### [NEW] `src/data/multimodal_dataset.py`
#### [NEW] `src/models/two_tower.py`

### 2. Large-Scale Training Pipeline
- Contrastive learning loop (In-Batch Negatives).
- PyTorch Lightning setup for multi-GPU scaling.
#### [NEW] `src/training/trainer.py`
#### [NEW] `src/training/loss.py` (InfoNCE loss)

### 3. Edge Optimization & Export
- Distillation and INT8 Quantization.
- Export to ONNX, CoreML (macOS/iOS), and TFLite.
#### [NEW] `src/deployment/quantize_export.py`

### 4. On-Device Retrieval Simulation
- Simulating the device-side logic: pushing pre-computed item embeddings to a local vector store, and running the exported User Model live.
#### [NEW] `src/deployment/edge_inference_sim.py`

## Verification Plan

### Automated Tests
- Unit tests verifying output tensor shapes of the multimodal fusion block.
- Verification that the ONNX/CoreML exported models produce the same embeddings (within a tolerance epsilon) as the native PyTorch models.

### Manual Verification
- We will run `edge_inference_sim.py` to timing the latency of generating a user embedding and executing a local vector search. The strict goal is `< 50ms` latency for the end-to-end on-device recommendation pass.
