import os
import random
import tempfile
import torch
import gradio as gr
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from hardware import detect_hardware, choose_model

HW = detect_hardware()
MODEL_CONFIG = choose_model(HW)
PIPE = None
LOAD_ERROR = None

def load_pipeline():
    global PIPE, LOAD_ERROR
    if PIPE is not None:
        return
    try:
        model_id = MODEL_CONFIG["model_id"]
        if MODEL_CONFIG["use_sdxl"]:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if HW["cuda_available"] else torch.float32,
                use_safetensors=True,
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if HW["cuda_available"] and HW["vram_gb"] >= 6 else torch.float32,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True,
            )

        if HW["cuda_available"]:
            pipe = pipe.to("cuda")
            # Memory-saving options; unsupported options are safely ignored.
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass
        else:
            pipe = pipe.to("cpu")
        PIPE = pipe
    except Exception as e:
        LOAD_ERROR = f"{type(e).__name__}: {e}"
        raise

def hardware_summary():
    return (
        f"**Detected hardware**\n\n"
        f"- GPU: `{HW['gpu_name'] or 'None detected'}`\n"
        f"- VRAM: `{HW['vram_gb']} GB`\n"
        f"- System RAM: `{HW['ram_gb'] if HW['ram_gb'] is not None else 'Unknown'} GB`\n"
        f"- CUDA: `{'available' if HW['cuda_available'] else 'not available'}`\n"
        f"- Selected model: **{MODEL_CONFIG['name']}** (`{MODEL_CONFIG['model_id']}`)\n\n"
        f"Model weights download on first generation. CPU-only generation may be extremely slow."
    )

def generate(prompt, negative_prompt, width, height, steps, guidance, seed):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")
    try:
        load_pipeline()
    except Exception as e:
        raise gr.Error(
            "Could not load the model. Check the Colab runtime, internet access, available disk/RAM, "
            f"and model availability. Details: {type(e).__name__}: {e}"
        )
    width, height = int(width), int(height)
    # Clamp dimensions to multiples of 8 and reasonable limits.
    width = max(256, min(1024, (width // 8) * 8))
    height = max(256, min(1024, (height // 8) * 8))
    steps = max(5, min(50, int(steps)))
    seed = int(seed)
    if seed < 0:
        seed = random.randint(0, 2**31 - 1)
    generator_device = "cuda" if HW["cuda_available"] else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)
    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=float(guidance),
        generator=generator,
    )
    if HW["cuda_available"]:
        with torch.inference_mode(), torch.autocast("cuda"):
            result = PIPE(**kwargs)
    else:
        with torch.inference_mode():
            result = PIPE(**kwargs)
    image = result.images[0]
    fd, path = tempfile.mkstemp(prefix="colab_ai_", suffix=".png")
    os.close(fd)
    image.save(path)
    return image, path, f"Seed: {seed} · Model: {MODEL_CONFIG['name']}"

with gr.Blocks(title="Colab AI Image Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 Colab AI Image Generator\nGenerate images with a model selected for this Colab runtime.")
    gr.Markdown(hardware_summary())
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder="A cozy cabin beside a lake at sunrise, cinematic lighting...", lines=4)
            negative = gr.Textbox(label="Negative prompt", placeholder="blurry, low quality, distorted anatomy...", lines=2)
            with gr.Row():
                width = gr.Slider(256, 1024, value=MODEL_CONFIG["width"], step=64, label="Width")
                height = gr.Slider(256, 1024, value=MODEL_CONFIG["height"], step=64, label="Height")
            with gr.Row():
                steps = gr.Slider(5, 50, value=MODEL_CONFIG["steps"], step=1, label="Inference steps")
                guidance = gr.Slider(1, 12, value=7, step=0.5, label="Guidance scale")
            seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
            run = gr.Button("Generate image", variant="primary")
        with gr.Column():
            output = gr.Image(label="Generated image", type="pil")
            download = gr.File(label="Download PNG")
            details = gr.Markdown()
    run.click(
        fn=generate,
        inputs=[prompt, negative, width, height, steps, guidance, seed],
        outputs=[output, download, details],
    )
    gr.Markdown("**Privacy note:** the temporary Gradio share link is public to anyone who has it while active. Do not enter secrets.")

if __name__ == "__main__":
    demo.queue().launch(share=True, debug=True)
