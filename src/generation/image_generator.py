import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, AutoPipelineForInpainting
import time
from PIL import Image, ImageDraw

class RetailImageGenerator:
    """
    RAG-driven Image Generation for Retail Products.
    Allows taking a recommended product (or its embeddings) and a user prompt
    to generate customized product imagery, or inpainting it into a user's space.
    """
    def __init__(self, mode="cloud"):
        self.mode = mode
        print(f"🚀 Initializing Generative AI Suite in [{self.mode.upper()}] mode...")
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        
        if self.mode == "cloud":
            model_id = "runwayml/stable-diffusion-v1-5"
            print(f"Loading Text-to-Image Pipeline ({model_id})...")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32, 
                requires_safety_checker=False,
                safety_checker=None
            )
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            self.pipe = self.pipe.to(self.device)
            self.pipe.enable_attention_slicing() 
            
            # Load Inpainting Pipeline (We use AutoPipelineForInpainting to handle inpainting tasks)
            # For optimal results, we use the specific inpainting checkpoint of SD 1.5
            inpaint_model_id = "runwayml/stable-diffusion-inpainting"
            print(f"Loading Inpainting Pipeline for Spatial Visualizations ({inpaint_model_id})...")
            self.inpaint_pipe = AutoPipelineForInpainting.from_pretrained(
                inpaint_model_id,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                requires_safety_checker=False,
                safety_checker=None
            )
            self.inpaint_pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.inpaint_pipe.scheduler.config)
            self.inpaint_pipe = self.inpaint_pipe.to(self.device)
            self.inpaint_pipe.enable_attention_slicing()

        elif self.mode == "edge":
            print("⚙️ Preparing Edge-Optimized Pipeline (Simulated)")
            model_id = "stabilityai/sd-turbo"  
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.float32, 
                requires_safety_checker=False,
                safety_checker=None
            )
            self.pipe = self.pipe.to(self.device)
            
            # Edge inpainting utilizes the same turbo backbone for extreme speed
            self.inpaint_pipe = AutoPipelineForInpainting.from_pipe(self.pipe)
            self.inpaint_pipe = self.inpaint_pipe.to(self.device)
            
    def generate_customized_product(self, base_product_name, user_custom_prompt):
        """
        Generates an image of the recommended product modified by the user's prompt.
        """
        final_prompt = f"Professional studio photography of {base_product_name}. {user_custom_prompt}, 8k resolution, photorealistic, highly detailed, clean white background"
        negative_prompt = "low resolution, ugly, blurry, text, watermark, bad anatomy"
        
        print(f"🎨 Generating customized product: '{final_prompt}'")
        start_time = time.time()
        num_inference_steps = 20 if self.mode == "cloud" else 4 
        
        with torch.no_grad():
            image = self.pipe(
                prompt=final_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5 if self.mode == "cloud" else 1.5,
                num_images_per_prompt=1
            ).images[0]
            
        print(f"✅ Generated in {time.time() - start_time:.2f} seconds!")
        return image

    def generate_inpaint_suggestion(self, base_image: Image.Image, mask_image: Image.Image, product_prompt: str):
        """
        Inpaints a suggested product into a user's contextual space 
        (e.g., placing a couch into an empty living room).
        """
        # Ensure images are properly sized and in RGB format for the diffusers pipeline
        base_image = base_image.convert("RGB").resize((512, 512))
        mask_image = mask_image.convert("RGB").resize((512, 512))
        
        final_prompt = f"A high quality photorealistic {product_prompt}, perfect lighting, seamlessly integrated into the room"
        negative_prompt = "low resolution, ugly, blurry, text, floating, badly cropped, bad anatomy"
        
        print(f"🛋️  Inpainting spatial visualization for: '{product_prompt}'")
        start_time = time.time()
        num_inference_steps = 25 if self.mode == "cloud" else 4
        
        with torch.no_grad():
            image = self.inpaint_pipe(
                prompt=final_prompt,
                negative_prompt=negative_prompt,
                image=base_image,
                mask_image=mask_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5 if self.mode == "cloud" else 1.5,
            ).images[0]
            
        print(f"✅ Spatial Inpainting completed in {time.time() - start_time:.2f} seconds!")
        return image


def create_mock_room_and_mask():
    """Helper function to generate a mock empty room and a center mask for testing."""
    room = Image.new("RGB", (512, 512), color="lightgray")
    draw = ImageDraw.Draw(room)
    # Draw some "walls" and "floor" to simulate a room
    draw.rectangle([0, 300, 512, 512], fill="burlywood") # Floor
    
    # Mask is white where we WANT to generate (e.g. where the couch goes)
    mask = Image.new("RGB", (512, 512), color="black")
    mask_draw = ImageDraw.Draw(mask)
    # Draw a rectangle in the center where the couch should be
    mask_draw.rectangle([100, 200, 412, 400], fill="white")
    
    return room, mask


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="cloud", choices=["cloud", "edge"])
    args = parser.parse_args()
    
    generator = RetailImageGenerator(mode=args.mode)
    
    # --- DEMO 1: Standard Product Customization ---
    print("\n--- DEMO 1: Prompt Customization ---")
    try:
        base_product = "A high quality canvas tote bag"
        user_prompt = "Dark moody floral pattern, embroidered details"
        img1 = generator.generate_customized_product(base_product, user_prompt)
        img1.save(f"customized_{args.mode}.png")
        print(f"💾 Saved product to customized_{args.mode}.png")
    except Exception as e:
        print(f"❌ Customization failed: {e}")

    # --- DEMO 2: Spatial Inpainting ---
    print("\n--- DEMO 2: Spatial Inpainting (Room Visualization) ---")
    try:
        user_room, placement_mask = create_mock_room_and_mask()
        recommended_couch = "modern mid-century brown leather couch with wooden legs"
        
        img2 = generator.generate_inpaint_suggestion(user_room, placement_mask, recommended_couch)
        img2.save(f"inpainted_room_{args.mode}.png")
        print(f"💾 Saved spatial visualization to inpainted_room_{args.mode}.png")
    except Exception as e:
        print(f"❌ Inpainting failed: {e}")
        print("Note: If running into memory issues, run on a system with 16GB+ RAM or a dedicated GPU.")
