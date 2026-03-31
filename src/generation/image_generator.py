import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
# Uncomment for IP-Adapter usage if you're conditioning heavily on the product image
# from diffusers import AutoPipelineForText2Image 
import time
from PIL import Image

class RetailImageGenerator:
    """
    RAG-driven Image Generation for Retail Products.
    Allows taking a recommended product (or its embeddings) and a user prompt
    to generate customized product imagery.
    """
    def __init__(self, mode="cloud"):
        """
        mode: 
          - 'cloud': Uses a larger model like SD-v1.5 or SDXL in FP16 precision
          - 'edge': Uses CoreML or Quantized ONNX for mobile device inference
        """
        self.mode = mode
        print(f"🚀 Initializing Image Generator in [{self.mode.upper()}] mode...")
        
        if self.mode == "cloud":
            # In a real cloud backend, you'd load SDXL or a fast variant like SD-Turbo
            model_id = "runwayml/stable-diffusion-v1-5"
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.float16, 
                requires_safety_checker=False,
                safety_checker=None
            )
            # Use faster solver
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            
            # If a LoRA is available, load it here
            # self.pipe.load_lora_weights("path/to/retail_lora", weight_name="pytorch_lora_weights.safetensors")
            
            # Moves to CUDA if available, else CPU/MPS
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            self.pipe = self.pipe.to(self.device)
            self.pipe.enable_attention_slicing() # Save VRAM
            
        elif self.mode == "edge":
            # For edge, you would typically use ANE (Apple Neural Engine) via coremltools
            # or an ONNX/TFLite optimized model for Android/iOS.
            # Here we simulate an ultra-fast Edge model setup like SD-Turbo or a distilled model:
            print("⚙️ Preparing Edge-Optimized Pipeline (Simulated)")
            model_id = "stabilityai/sd-turbo"  # Extremely fast, 1-4 step generation
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.float32, 
                requires_safety_checker=False,
                safety_checker=None
            )
            self.device = "cpu"
            self.pipe = self.pipe.to(self.device)
            # CoreML/ONNX exports would happen here natively in a mobile app
            
    def generate_customized_product(self, base_product_name, user_custom_prompt):
        """
        Generates an image of the recommended product modified by the user's prompt.
        """
        # Construct the final prompt natively using the Recommended Product's attributes
        # and the user's desired modifications.
        # E.g. base_product_name = "Nike Air Max sneakers"
        #      user_custom_prompt = "Make it bright blue with neon green laces"
        final_prompt = f"Professional studio photography of {base_product_name}. {user_custom_prompt}, 8k resolution, photorealistic, highly detailed, clean white background"
        negative_prompt = "low resolution, ugly, blurry, text, watermark, bad anatomy"
        
        print(f"🎨 Generating image for prompt: '{final_prompt}'")
        
        start_time = time.time()
        
        # Generation steps vary based on edge vs cloud
        num_inference_steps = 20 if self.mode == "cloud" else 4 # Turbo needs fewer steps
        
        with torch.no_grad():
            image = self.pipe(
                prompt=final_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5 if self.mode == "cloud" else 1.5, # Turbo uses low guidance
                num_images_per_prompt=1
            ).images[0]
            
        latency = time.time() - start_time
        print(f"✅ Image generated in {latency:.2f} seconds!")
        
        return image

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="cloud", choices=["cloud", "edge"])
    args = parser.parse_args()
    
    # 1. Initialize our Generator
    generator = RetailImageGenerator(mode=args.mode)
    
    # 2. Simulate User Interaction
    # The RecSys pipeline recommended a "Canvas Tote Bag"
    base_product = "A high quality canvas tote bag"
    # The user taps "Customize" and types: "Make it a dark floral pattern"
    user_prompt = "Dark moody floral pattern, embroidered details"
    
    # 3. Generate Image
    try:
        generated_img = generator.generate_customized_product(base_product, user_prompt)
        generated_img.save(f"customized_product_{args.mode}.png")
        print(f"💾 Saved customized product to customized_product_{args.mode}.png")
    except Exception as e:
        print(f"❌ Failed to generate image: {e}")
        print("Note: If running into memory issues or missing network access for Huggingface, ensure diffusers is installed.")
