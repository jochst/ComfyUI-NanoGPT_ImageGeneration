# NanoGPT ImageGeneration
Custom node for NanoGPT image generation, use directly on ComfyUI or via SillyTavern, OpenWebUI etc.<br />
I made this to use the image models included on 8$ plan, but may work with other models too.

# Install
1. Install the node
```
cd ComfyUI/custom_nodes/
git clone https://github.com/myonmu0/ComfyUI-NanoGPT_ImageGeneration
```

2. Edit nodes.py and put your NanoGPT api key as follow:
```
API_KEY = "sk-nano-XXXX-XXXX-XXXX-XXXX-XXXX"
```

# Usage
1. **Image Generation:**
Load examples/workflow.json to ComfyUI, select the model and run, to use in SillyTavern or OpenWebUI etc, on ComfyUI go to Menu > File > Export (API) and load that json on your UI.
![example](https://github.com/myonmu0/ComfyUI-NanoGPT_ImageGeneration/blob/main/examples/1.png)

2. **Image Edit:**
Use qwen-image model, Image Edit won't work on z-image-turbo, chroma or hidream.<br />
Also this works on OpenWebUI.<br />
![example](https://github.com/myonmu0/ComfyUI-NanoGPT_ImageGeneration/blob/main/examples/2.png)


