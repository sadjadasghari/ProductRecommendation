import urllib.request
import urllib.error
import base64

mermaid_code = """flowchart TB
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
    
    ModelArchitecture == "Post-Training Quantization (INT8) + ONNX" ==> UserTower
    ModelArchitecture == "Export FP32 Weights" ==> ItemTower

    %% Gen AI Node
    subgraph Generative ["GenAI Customization Pipeline"]
        direction TB
        RAG["Base Product Context"]
        TextPrompt["User Prompt"]
        UserRoom["User Photo (Living Room)"]
        RoomMask["Target Placement Mask"]
        
        Diffusion("Stable Diffusion (Txt2Img)")
        Inpaint("Stable Diffusion (Inpainting)")
        
        FinalImage["Customized Product Image"]
        SpatialImage["Spatial Visualization (Couch in Room)"]
        
        RAG --> Diffusion
        TextPrompt --> Diffusion
        Diffusion --> FinalImage
        
        TextPrompt --> Inpaint
        UserRoom --> Inpaint
        RoomMask --> Inpaint
        Inpaint --> SpatialImage
    end

    Recommendations -. "User taps Customize" .-> RAG
    Recommendations -. "User taps View in Room" .-> UserRoom
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
