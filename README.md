# Colab AI Image Generator

A lightweight GitHub repository that launches an AI image-generation UI in Google Colab using Gradio.

## Features
- Detects GPU model, VRAM, system RAM, and CUDA availability.
- Automatically selects a model based on available VRAM:
  - **12 GB VRAM or more:** SDXL Base 1.0
  - **6–12 GB VRAM:** Stable Diffusion 1.5
  - **Less than 6 GB VRAM / CPU-only:** Stable Diffusion 1.5 with conservative settings (may be very slow or fail on tight RAM)
- Gradio UI for prompt, negative prompt, dimensions, inference steps, guidance scale, and seed.
- Generated images can be downloaded directly from the Gradio result.
- No model weights are stored in this repository; models are downloaded at runtime.

## Run in Google Colab
1. Upload this repository to your GitHub account, or use the notebook in `AI_Image_Generator.ipynb`.
2. Open the notebook in Colab.
3. In **Runtime → Change runtime type**, select a GPU runtime if available.
4. Run the cells from top to bottom.
5. Use the Gradio UI shown at the bottom of the notebook.

## Open the notebook directly from GitHub
After publishing the repository, replace `YOUR_GITHUB_USERNAME` below with your GitHub username:

`https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/colab-ai-image-generator/blob/main/AI_Image_Generator.ipynb`

## Notes
- Colab free GPU availability, runtime duration, and memory are controlled by Google and are not guaranteed.
- A public Gradio share URL is temporary and accessible to anyone who has it while the session is active. Do not enter private prompts or expose secrets.
- Model downloads require internet access and consume runtime disk space.
- Model licenses and acceptable-use terms are set by their publishers. Review them before commercial use.
- The automatic selector is a heuristic, not a guarantee. Free VRAM and system RAM can change during a session.
