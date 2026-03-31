# Industrial Architecture Modernization Plan

## Goal
To elevate the Multimodal Edge RecSys pipeline to a state-of-the-art enterprise standard. We will implement Vision Transformers, HNSW Vector Databases, Hard Negative Contrastive Loss, a Feature Store integration pattern, and an Agentic LLM router. Finally, we will overhaul the presentation slides and architecture diagram to reflect these cutting-edge additions.

## User Review Required
> [!IMPORTANT]
> This plan touches almost every component of the system. 
> 
> 1.  **Vision Backbone:** I will swap `MobileNetV3` for a lightweight **Vision Transformer (`vit_b_16` or similar)**. ViTs are heavier; do you want to keep the Cloud `Item Tower` as heavy as needed while prioritizing Edge compression specifically for the `User Tower`?
> 2.  **Retrieval Engine:** I will add `faiss-cpu` to the environment and rebuild the `evaluate.py` ranking logic to leverage highly optimized Approximate Nearest Neighbor (ANN) indexing rather than brute-force exact matrix multiplication.
> 3.  **Are you okay with proceeding with all 5 of these major upgrades?**

## Proposed Changes

### Component 1: Data & Modeling (ViT & Context)
#### [MODIFY] [multimodal_dataset.py](file:///Users/s0a0dhl/Workspace/ProductRecommendation/src/data/multimodal_dataset.py)
*   Add mock "real-time context" features (e.g., time of day, location type) simulating a **Feature Store** ingestion.
#### [MODIFY] [two_tower.py](file:///Users/s0a0dhl/Workspace/ProductRecommendation/src/models/two_tower.py)
*   `ItemEncoder`: Replace `models.mobilenet_v3_small` with `models.vit_b_16` (Vision Transformer).
*   `UserEncoder`: Add a secondary injection layer to concatenate the Feature Store user-context before passing through the final projection MLP.

### Component 2: Advanced Contrastive Loss
#### [MODIFY] [loss.py](file:///Users/s0a0dhl/Workspace/ProductRecommendation/src/training/loss.py)
*   Upgrade standard InfoNCE to **Hard Negative InfoNCE**, optionally with additive margin concepts, by adjusting the temperature scaling and mining the hardest samples per batch. This forces the model to better distinguish visually similar but semantically distinct products.

### Component 3: HNSW Retrieval Infrastructure
#### [NEW] [requirements.txt](file:///Users/s0a0dhl/Workspace/ProductRecommendation/requirements.txt) (Update)
*   Add `faiss-cpu` for ANN indexing.
#### [MODIFY] [evaluate.py](file:///Users/s0a0dhl/Workspace/ProductRecommendation/src/evaluation/evaluate.py)
*   Remove exact dense retrieval `torch.matmul()`.
*   Build a **FAISS `IndexHNSWFlat`** with the 128-dim item embeddings.
*   Query user embeddings against the FAISS index to compute HR@10, proving O(log N) scale retrieval.

### Component 4: Agentic AI Orchestrator
#### [NEW] [agent_router.py](file:///Users/s0a0dhl/Workspace/ProductRecommendation/src/generation/agent_router.py)
*   Mock an **LLM Orchestrator** class. It accepts raw user text like *"I need a red shirt"* or *"Does this couch fit my room (photo attached)?"* and automatically dynamically routes between the `TwoTowerRecSys` retrieval API and the `StableDiffusionInpaintPipeline`. 

### Component 5: Presentation & Documentation
#### [MODIFY] [architecture_diagram.md](file:///Users/s0a0dhl/.gemini/antigravity/brain/d5312596-427d-43b4-ba15-39ccf6afa5e1/architecture_diagram.md)
*   Redraw the Mermaid chart completely. Add Feature Store node, FAISS Node, ViT icon, and the LLM Orchestrator router.
#### [MODIFY] [presentation_slides.md](file:///Users/s0a0dhl/.gemini/antigravity/brain/d5312596-427d-43b4-ba15-39ccf6afa5e1/presentation_slides.md)
*   Update slide content explicitly referencing ViT, HNSW, Feature Stores, Hard Negative Loss, and LLM Agents for the Apple presentation.

## Verification Plan

### Automated Tests
1. `python -m src.training.trainer` to ensure ViT and Feature Store Context properly flow through the modified Custom Transformer and Hard Negative Loss without shape mismatch.
2. `pip install faiss-cpu` and `python -m src.evaluation.evaluate` to ensure FAISS HNSW graph builds correctly and searches the top-10 items locally.
3. Rerun `generate_presentation.py` to compile the final interview artifact `.pptx`.
