# Model Evaluation Implementation Plan

This plan details the steps to evaluate the recommendation quality of the Two-Tower model trained for the Multimodal Edge RecSys project. We will compute standard ranking metrics (HR@K, NDCG@K) for both the FP32 and INT8 quantized models to assess the trade-off between quantization and recommendation accuracy.

## Proposed Changes

### Evaluation Module
We will create a new evaluation script to compute metrics and compare models.

#### [NEW] [evaluate.py](file:///Users/s0a0dhl/Workspace/ProductRecommendation/src/evaluation/evaluate.py)
This script will:
1. Load a test set using the existing [MultimodalRetailDataset](file:///Users/s0a0dhl/Workspace/ProductRecommendation/src/data/multimodal_dataset.py#5-41).
2. Generate user embeddings and item embeddings.
3. Compute the cosine similarity between all user embeddings and item embeddings.
4. For each user $i$, its ground truth positive item is item $i$ in the batch. We evaluate the rank of item $i$ among all items.
5. Compute **Hit Rate (HR@10)** and **NDCG@10**.
6. Compare the metrics between:
   - The FP32 [UserEncoder](file:///Users/s0a0dhl/Workspace/ProductRecommendation/src/models/two_tower.py#57-89) ([edge_user_model_fp32.pt](file:///Users/s0a0dhl/Workspace/ProductRecommendation/edge_user_model_fp32.pt)).
   - The INT8 quantized [UserEncoder](file:///Users/s0a0dhl/Workspace/ProductRecommendation/src/models/two_tower.py#57-89) ([edge_user_model_int8.pt](file:///Users/s0a0dhl/Workspace/ProductRecommendation/edge_user_model_int8.pt)).
   - The original PyTorch Lightning [TwoTowerRecSys](file:///Users/s0a0dhl/Workspace/ProductRecommendation/src/models/two_tower.py#91-107).

## Verification Plan

### Automated Tests
1. Run `python -m src.evaluation.evaluate` from the project root.
2. Verify that the script outputs HR@10 and NDCG@10 for both the FP32 and INT8 models.
3. Validate that the metrics are within a reasonable range (since it's an untrained/fast-dev-run model, the metrics might be random, but the calculation logic must be correct).

### Manual Verification
1. Review the generated logs in the console to ensure both FP32 and INT8 models are successfully loaded and evaluated without errors.
2. Ensure the latency or accuracy drop of the INT8 model is reported in the console.
