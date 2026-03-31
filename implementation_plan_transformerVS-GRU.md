# Transformer Upgrade Implementation Plan

## Goal
To deprecate the `GRU` text backbone in the **Item Tower** and replace it with a state-of-the-art Transformer architecture. 

A standard recurrent model like a GRU processes text left-to-right natively, which is suboptimal for rich product descriptions. A Transformer, however, uses Self-Attention to capture bidirectional context across the entire product description simultaneously, vastly improving the quality of the multimodal `Item Embedding`.

## User Review Required
> [!IMPORTANT]
> To execute this upgrade, I propose we migrate the text backbone from a trained-from-scratch GRU to a pre-trained foundation model, specifically **`sentence-transformers/all-MiniLM-L6-v2`** or **`distilbert-base-uncased`**. 
> 
> *   Do you want to use the HuggingFace `transformers` library for the text backbone? It's industry-standard for this context.
> *   *Note*: The `User Tower` (which models the dynamic sequence of items) currently also uses a GRU. Do you want to upgrade the *User Tower* to a Transformer as well (e.g., SASRec - Self-Attentive Sequential Recommendation), or just strictly upgrade the Text descriptions in the Item Tower?

## Proposed Changes

### 1. Data Pipeline Updates
#### [MODIFY] [multimodal_dataset.py](file:///Users/s0a0dhl/Workspace/ProductRecommendation/src/data/multimodal_dataset.py)
*   Integrate a HuggingFace `AutoTokenizer`. Instead of generating purely random integer arrays for text tokens, the `__getitem__` function will output strictly formatted Transformer-ready `input_ids` and `attention_mask` tensors describing mock retail items.

### 2. Model Architecture Upgrade
#### [MODIFY] [two_tower.py](file:///Users/s0a0dhl/Workspace/ProductRecommendation/src/models/two_tower.py)
*   Remove the `nn.Embedding` and `nn.GRU` layers inside the `ItemEncoder`.
*   Inject an `AutoModel` (DistilBERT/MiniLM).
*   Extract the `[CLS]` token embedding (or apply Mean Pooling over the LAST hidden state using the attention mask) representing the textual semantic essence of the product.
*   Fuse this output (typically 384 or 768 dimensions) with the 576-dim `MobileNetV3` features.

### 3. Training & Evaluation Adjustments
#### [MODIFY] [trainer.py](file:///Users/s0a0dhl/Workspace/ProductRecommendation/src/training/trainer.py)
#### [MODIFY] [evaluate.py](file:///Users/s0a0dhl/Workspace/ProductRecommendation/src/evaluation/evaluate.py)
*   Ensure the forward passes provide the `attention_mask` required by the Transformer.

## Verification Plan

### Automated Tests
1. Run `python -m src.training.trainer` to verify the new Transformer backbone successfully compiles, passes embeddings, and computes the contrastive InfoNCE loss seamlessly without OOM (Out-of-Memory) errors.
2. Ensure the fusion mechanism cleanly concatenates the Transformer output with the Vision output.
