# NanoGPT ImageGeneration
Custom node for NanoGPT image generation, use directly on ComfyUI or via SillyTavern, OpenWebUI etc.<br />
I made this to use the image models included on 8$ plan, but may work with other models too.

# Install
1. Install the node
```
cd ComfyUI/custom_nodes/
git clone https://github.com/myonmu0/ComfyUI-NanoGPT_ImageGeneration
```

2. Put your NanoGPT API key in `api_key.txt` in this folder (bare key), or set the `NANOGPT_API_KEY` env var.

# Nodes
- **NanoGPT Image (NSFW, paid)**: NSFW-capable pay-per-use image models (live list, fallback `models_snapshot.json`).
- **NanoGPT Video (NSFW, paid)**: NSFW-capable pay-per-use video models. Outputs VIDEO + URL.
- **NanoGPT Prompt Generator**: chat-completion call that expands an idea into a prompt; wire its output to the image/video prompt.
