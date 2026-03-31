import collections 
import collections.abc
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN

def create_slide(prs, title, content):
    slide_layout = prs.slide_layouts[1] # Title and Content
    slide = prs.slides.add_slide(slide_layout)
    title_shape = slide.shapes.title
    body_shape = slide.shapes.placeholders[1]
    
    title_shape.text = title
    
    tf = body_shape.text_frame
    tf.clear()
    
    for i, bullet in enumerate(content):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = bullet
        p.font.size = Pt(24)

def generate_presentation():
    # Fix dict attribute error sometimes seen in older pptx versions with python 3.10+
    prs = Presentation()
    
    # Title Slide
    title_slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(title_slide_layout)
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    
    title.text = "Multimodal Intelligent System Experiences (ISE)"
    subtitle.text = "A Privacy-First Architecture for Personalized Product Discovery\nCandidate Presentation: Manager, SIML"
    
    # Slides Content
    slides_data = [
        {
            "title": "1. Vision & Strategy (SIML Focus)",
            "content": [
                "Goal: Deliver highly personalized, multimodal intelligent experiences natively on Apple devices.",
                "System Intelligence: Decoupling heavy representation learning from on-device sequential modeling.",
                "Privacy-First: User click streams remain entirely on-device.",
                "Scalability: Designed to scale across Apple's ecosystem (iOS, iPadOS, macOS) via CoreML and the Apple Neural Engine (ANE)."
            ]
        },
        {
            "title": "2. Multimodal Foundation Models",
            "content": [
                "Catalog Embedding Cloud: Pre-processing the product index purely offline.",
                "Vision Backbone: MobileNetV3 (or proprietary vision models) for feature extraction.",
                "Text Backbone: Fused language encoders for rich product descriptions.",
                "Result: A unified 128-dimensional multimodal item embedding representing complex semantics."
            ]
        },
        {
            "title": "3. Efficient On-Device ML Architecture",
            "content": [
                "RecSys Edge Tower: A fast sequential GRU architecture processing local interaction history.",
                "Contrastive Training (InfoNCE): Trained symmetrically for strong semantic alignment between intent and catalog.",
                "Post-Training Quantization (PTQ): Reduced memory footprint by ~74% (from FP32 to dynamic INT8) with zero degradation in NDCG.",
                "Inference: Executing natively via ANE/CPU with strict latency constraints (< 2ms)."
            ]
        },
        {
            "title": "4. Personalized Generative AI (RAG + Diffusion)",
            "content": [
                "User Empowerment: Seamless transition from Recommendation to Creation.",
                "Diffusion Integration: Using IP-Adapter and distilled models (e.g., SD-Turbo) to generate customized products.",
                "Edge Generation: Optimizing diffusion for local ANE execution (< 4 steps) to preserve privacy and reduce server costs.",
                "LoRA Fine-Tuning: Applying Low-Rank Adaptation to strictly adhere to brand guidelines and style aesthetics."
            ]
        },
        {
            "title": "5. End-to-End Evaluation Framework",
            "content": [
                "Recommendation Quality: Driving engagement via Hit Rate (HR@10) and NDCG.",
                "Generative AI Metrics: Automated CLIP Score evaluation for text-to-image alignment and prompt adherence.",
                "Continuous Improvement: Setting up infrastructure for RLHF and online A/B testing (CTR tracking)."
            ]
        },
        {
            "title": "6. Team Leadership & Roadmap",
            "content": [
                "Cross-Functional Execution: Bridging researchers, mobile developers, and product teams.",
                "Infrastructure: Standardizing on scalable tools (PyTorch Lightning, Ray Serve) vs Edge deployments (CoreML).",
                "Next Steps: Exploring LLM integration for conversational product search and expanding Multimodal Foundation capabilities."
            ]
        }
    ]
    
    for slide_data in slides_data:
        create_slide(prs, slide_data["title"], slide_data["content"])
        
    prs.save('apple_siml_interview_presentation.pptx')
    print("✅ Successfully generated 'apple_siml_interview_presentation.pptx'")

if __name__ == "__main__":
    generate_presentation()
