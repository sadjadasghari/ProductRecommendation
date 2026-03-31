# Multimodal Edge RecSys: System Architecture

The following Mermaid diagram illustrates the end-to-end Machine Learning System Design pipeline, detailing the separation of concerns between Cloud precomputation, Edge inference, PyTorch Model Training, and Generative AI integration natively tailored for an ISE (Intelligent System Experience).

```mermaid
flowchart TB
    %% Definitions
    classDef edge fill:#f9f2ec,stroke:#d78a1a,stroke-width:2px;
    classDef cloud fill:#e6f3ff,stroke:#267326,stroke-width:2px;
    classDef training fill:#f0e6ff,stroke:#5c00e6,stroke-width:2px;
    classDef genai fill:#ffffe6,stroke:#b3b300,stroke-width:2px;

    %% Client Node
    subgraph Device ["Mobile Edge Device"]
        History["Local Click History"]
        UserTower("User Tower GRU")
        LocalDB[("Local Vector Index")]
        Recommendations["Ranked UI"]
        
        History --> |"128-dim vectors"| UserTower
        UserTower --> |"Generates Target Vector"| LocalDB
        LocalDB --> |"Cosine Sim Top-K"| Recommendations
    end

    %% Cloud Node
    subgraph Backend ["Cloud Infrastructure"]
        Catalog(["Raw Catalog: Images & Text"])
        ItemTower("Item Tower MobileNetV3 + GRU")
        CloudDB[("Global Vector DB")]
        
        Catalog --> ItemTower
        ItemTower --> |"Generates Item Embeddings"| CloudDB
    end
    
    %% Sync Path
    CloudDB -. "Periodic Catalog Sync" .-> LocalDB

    %% Training Node
    subgraph TrainingPipeline ["Distributed Training Loop"]
        Dataset("Multimodal Dataset")
        ModelArchitecture{"Two-Tower Structure"}
        Loss("Symmetric InfoNCE Loss")
        
        Dataset --> ModelArchitecture
        ModelArchitecture --> Loss
        Loss -. "Optimize" .-> ModelArchitecture
    end
    
    %% Deployment Paths
    ModelArchitecture == "Post-Training Quantization (INT8) + ONNX" ==> UserTower
    ModelArchitecture == "Export FP32 Weights" ==> ItemTower

    %% Gen AI Node
    subgraph Generative ["GenAI Customization Pipeline"]
        RAG["Base Product Context"]
        TextPrompt["User Prompt"]
        Diffusion("Stable Diffusion Pipeline")
        FinalImage["Customized Product Image"]
        
        RAG --> Diffusion
        TextPrompt --> Diffusion
        Diffusion --> FinalImage
    end

    %% Connect UI to GenAI
    Recommendations -. "User taps Customize" .-> RAG
```

### Component Details
1. **Mobile Edge Device (Orange):** Represents the on-device execution environment. The `User Tower` handles sequential data. Privacy is maximally protected because the user's `Click History` never transits the network.
2. **Cloud Infrastructure (Green):** Represents the batch ingestion of new catalog items, offloading heavy Multi-modal encoding (Vision+Text representation) to the cloud.
3. **Training Pipeline (Purple):** Represents the shared InfoNCE modeling phase integrating PyTorch Lightning for cluster training.
4. **GenAI Pipeline (Yellow):** Represents the post-recommendation RAG interface letting users synthesize completely novel products from existing ones using fine-tuned Diffusion models.
