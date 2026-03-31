# Inpainting Functionality Implementation Plan

## Goal
To allow users to upload personal context photos (e.g., a photo of their living room) and use **Image Inpainting** to seamlessly generate and visualize product recommendations (e.g., a couch) directly inside their space, based on their prompts.

## User Review Required
> [!IMPORTANT]
> This requires adding a new `StableDiffusionInpaintPipeline` to our Generative AI component. Because loading two heavy pipelines (Text-to-Image and Inpainting) simultaneously can exceed VRAM limits on edge devices or standard cloud GPUs, I propose we lazy-load the pipelines or use an **AutoPipeline** that shares identical base weights between text-to-image and inpainting. Please confirm if this is acceptable!

## Proposed Changes

### Generative AI Pipeline

#### [MODIFY] [image_generator.py](file:///Users/s0a0dhl/Workspace/ProductRecommendation/src/generation/image_generator.py)
*   **Initialization**: Import `StableDiffusionInpaintPipeline` from `diffusers`. Optionally instantiate it alongside the text-to-image pipeline for Cloud generation modes using `"runwayml/stable-diffusion-inpainting"`.
*   **New Method `generate_inpaint_suggestion()`**:
    *   **Inputs**: `base_image` (the user's living room), `mask_image` (the area to generate the new product), `prompt` (the product description).
    *   **Logic**: Process the base and mask images, generate the localized inpainting using the prompt (e.g., "A modern brown leather couch"), and return the composited room image.
*   **Demo Update**: Update the `__main__` execution block to demonstrate the inpainting flow by creating dummy "living room" and "mask" images for an automated functional test.

### Documentation Updates

#### [MODIFY] [recsys_walkthrough.md](file:///Users/s0a0dhl/Workspace/ProductRecommendation/recsys_walkthrough.md)
*   Append details on the **GenAI Inpainting** feature under the Generative AI Integration section, explaining how it enables Spatial Product Visualizations.

#### [MODIFY] [architecture_diagram.md](file:///Users/s0a0dhl/.gemini/antigravity/brain/d5312596-427d-43b4-ba15-39ccf6afa5e1/architecture_diagram.md)
*   Add the "Inpainting & Masking" flow into the `Generative AI` subgraph to accurately reflect the system architecture.

## Verification Plan

### Automated Tests
1. Run `python -m src.generation.image_generator --mode cloud` 
2. Verify that the script successfully initializes the Inpainting Pipeline.
3. Verify that the simulation executes `generate_inpaint_suggestion` by passing mock images, generating the result, and saving it as `inpainted_living_room.png`.

### Manual Verification
1. Visually verify the code handles `base_image` and `mask_image` logic correctly for the HuggingFace `diffusers` library requirements (converted to RGB, proper PIL formats).
