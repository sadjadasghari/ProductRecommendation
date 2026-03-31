````carousel
# Multimodal Edge RecSys
**Next-Generation, Privacy-First Product Recommendation & Customization**

*An end-to-end ML System Design Pipeline Overview*
<!-- slide -->
# 1. Business Objectives & Requirements

### The Goal
Deliver highly personalized product recommendations with zero network latency, while empowering users to visually customize products using Generative AI. 

### Core Requirements
**Functional Requirements:**
* Recommend items based on sequential user history.
* Support multimodal items (Image + Text descriptions).
* Allow text-prompted product customization (GenAI).

**Non-Functional Requirements:**
* **Strict Latency:** Sub-50ms inference time for instant UI scrolling.
* **Privacy:** User interaction history must remain on-device.
* **Footprint:** The on-device model must be small enough (< 5MB) to not bloat the mobile app size.
<!-- slide -->
# 2. Data Pipeline Strategy

To handle the scale of a retail catalog while retaining rich representations, we decoupled the item processing from the user interaction loop.

* **Item Catalog (Multimodal):** Pre-computed catalog data consisting of high-res images and descriptive text tokens.
* **User History:** Sequences built dynamically on the mobile app capturing the last N product embeddings the user viewed/clicked in their current session.
* **Simulated Training Data:** Built a PyTorch `Dataset` that generates `Batch[UserHistory, TargetImage, TargetText]` to simulate sequential clickstreams.
<!-- slide -->
# 3. Model Architecture (Two-Tower)

We utilized a **Two-Tower Architecture**, splitting the workload between the Cloud and the Edge:

### 1. Item Tower (Cloud)
* **Vision Backbone:** `MobileNetV3` for extracting visual features.
* **Text Backbone:** `GRU`-based sequence encoder for product descriptions.
* **Fusion:** Concatenates both streams into a unified dense embedding (128-dim).

### 2. User Tower (Edge)
* **Sequential Encoding:** A lightweight `GRU` processes the sequence of previous item embeddings.
* **Prediction:** Outputs a 128-dim vector representing the user's *current* intent, mapped directly into the same vector space as the items.
<!-- slide -->
# 4. Model Training Pipeline

Training requires bridging the Item Tower and User Tower into a shared semantic space.

* **Objective:** Contrastive Learning using **Symmetric InfoNCE Loss** (inspired by CLIP).
* **Mechanics:** Maximizes the cosine similarity between a User's embedding and their actual true next-clicked Item embedding, while minimizing similarity against all other "in-batch negative" items.
* **Framework:** **PyTorch Lightning** for scalable, fault-tolerant distributed training across GPU clusters, utilizing `AdamW` optimizers and Cosine Annealing learning rate schedules.
<!-- slide -->
# 5. Model Evaluation Metrics

Evaluation is split into Recommendation Quality and Generation Quality.

### Recommendation Evaluation (Offline)
* **Hit Rate @ 10 (HR@10):** Does the true target item appear in the top 10 retrieved items?
* **NDCG @ 10:** Are the most relevant items ranked higher in that top 10 list?
*(We benchmarked both the FP32 and INT8 models to ensure compression didn't degrade mathematical accuracy).*

### Generative Evaluation
* **CLIP Score:** Measures the cosine similarity between the final generated image and the user's customization prompt, ensuring prompt adherence.
* **Online A/B Testing:** Click-Through Rate (CTR) and Conversion Rate (CVR) for customized vs. static products.
<!-- slide -->
# 6. Edge Optimization

Recommender models are notoriously memory-heavy. To deploy our User Tower to mobile devices:

* **Post-Training Dynamic Quantization (PTQ):** We converted the PyTorch FP32 weights for `nn.Linear` and `nn.GRU` layers down to **INT8**. 
* **Compression Results:** 
  * Original FP32 Model: `1.32 MB`
  * Quantized INT8 Model: `0.35 MB`
  * **Reduction:** `~74%` smaller memory footprint with identical HR@10 on evaluation.
<!-- slide -->
# 7. Deployment & Inference Architecture

* **The Export:** We traced the INT8 quantized PyTorch model and exported it to **ONNX**. 
* **Mobile Wrappers:** ONNX allows native execution on iOS (via CoreML wrappers) and Android (via XNNPACK/QNN execution providers).
* **The Retrieval System:** The device maintains a localized vector database (e.g., SQLite-VSS or FAISS) of the current store catalog. 
* **Benchmark:** The end-to-end edge pipeline runs vector generation + nearest neighbor lookup consistently at **~1.5 - 2.0 ms** per batch, easily clearing the 50ms budget.
<!-- slide -->
# 8. Generative AI Integration (RAG + Diffusion)

We evolved the RecSys pipeline into a highly interactive GenAI experience.

### Architecture
When a user wants to customize a recommended product (e.g., "make it denim"):
1. The Edge Model predicts the desired item.
2. The user's prompt is passed to a **Diffusion Model** (Stable Diffusion v1.5 / SDXL).
3. **LoRA Fine-Tuning:** The diffusion model has been fine-tuned using `accelerate` on our retail catalog, acting as a style-guard to guarantee the generated images match our specific brand photography aesthetics.

### Deployment Tiers
* **Cloud Mode:** High-res generation (FP16) sitting behind Ray Serve/vLLM.
* **Edge Mode (Privacy):** Ultra-fast 4-step generation using distilled models like `SD-Turbo` running directly on the Apple Neural Engine.
<!-- slide -->
# 9. Conclusion & Business Impact

This pipeline redefines retail personalization:

1. **Zero-Lag UX:** By running the dynamic inference step purely on-device, the UI never blocks on a network request while scrolling.
2. **Absolute Privacy:** User click-history never leaves their phone; only randomized item embeddings exist in the global space.
3. **High Engagement:** Empowering users to iterate on recommendations with Generative AI directly drives product discovery and increases time-in-app, converting standard shopping into an interactive styling session.
````
