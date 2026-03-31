import urllib.request
import urllib.error
import base64

mermaid_code = """flowchart TB
    %% Intelligent Orchestrator
    subgraph Orchestrator ["Multimodal Agent Route"]
        LLM("LLM Orchestrator (Intent Router)")
    end

    %% Client Node
    subgraph Device ["Mobile Edge Device"]
        History["Local Click History"]
        FeatureStore[("Real-Time Feature Store (Context)")]
        UserTower("User Tower (SASRec Transformer)")
        LocalDB[("Local FAISS HNSW Index")]
        Recommendations["Ranked UI"]
        
        History --> |"128-dim vectors"| UserTower
        FeatureStore -. "Location, Time, Device" .-> UserTower
        UserTower --> |"Generates Target Vector"| LocalDB
        LocalDB --> |"O(log N) ANN Search"| Recommendations
    end

    %% Cloud Node
    subgraph Backend ["Cloud Infrastructure"]
        Catalog(["Raw Catalog: Images & Text"])
        ItemTower("Item Tower (ViT-B/16 + MiniLM Transf.)")
        CloudDB[("Global Vector DB")]
        
        Catalog --> ItemTower
        ItemTower --> |"Generates Item Embeddings"| CloudDB
    end

    CloudDB -. "Periodic Catalog Sync" .-> LocalDB

    %% Training Node
    subgraph TrainingPipeline ["Distributed Training Loop"]
        Dataset("Multimodal Dataset")
        ModelArchitecture{"Two-Tower Structure"}
        Loss("Hard Negative Contrastive Loss (Margin)")
        
        Dataset --> ModelArchitecture
        ModelArchitecture --> Loss
        Loss -. "Optimize" .-> ModelArchitecture
    end
    
    ModelArchitecture == "Post-Training Quantization (INT8) + ONNX" ==> UserTower
    ModelArchitecture == "Export FP32 Weights" ==> ItemTower

    %% Gen AI Node
    subgraph Generative ["GenAI Customization Pipeline"]
        direction TB
        RAG["Base Product Context"]
        TextPrompt["User Prompt"]
        UserRoom["User Photo (Living Room)"]
        RoomMask["Target Placement Mask"]
        
        Inpaint("Stable Diffusion (Inpainting)")
        
        FinalImage["Spatial Visualization (Couch in Room)"]
        
        TextPrompt --> Inpaint
        UserRoom --> Inpaint
        RoomMask --> Inpaint
        RAG --> Inpaint
        Inpaint --> FinalImage
    end

    %% Routing Flow
    LLM -. "Intent: Search" .-> History
    LLM -. "Intent: Visualize" .-> TextPrompt
    Recommendations -. "User Requests GenAI" .-> LLM
"""

encoded_str = base64.b64encode(mermaid_code.encode('utf-8')).decode('ascii')
url = f"https://mermaid.ink/img/{encoded_str}?type=png"
print(f"Fetching from {url}")

req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

try:
    with urllib.request.urlopen(req) as response:
        if response.status == 200:
            with open("architecture.png", "wb") as f:
                f.write(response.read())
            print("Success: Generated architecture.png")
except urllib.error.URLError as e:
    print(f"Failed to generate diagram image: {e}")
