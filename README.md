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
Load examples/workflow.json to ComfyUI, select the model and run.
![example](https://github.com/myonmu0/ComfyUI-NanoGPT_ImageGeneration/blob/main/examples/1.png)

For SillyTavern or OpenWebUI etc, to get random seed and yeld different image on each generation, load the examples/workflow_Random.json on ComfyUI, then go Menu > File > Export (API), and load that json to SillyTavern/OpenWebUI.
![example](https://github.com/myonmu0/ComfyUI-NanoGPT_ImageGeneration/blob/main/examples/1_2.png)

2. **Image Edit:**
Use qwen-image model, Image Edit won't work on z-image-turbo, chroma or hidream.<br />
Also this works on OpenWebUI, to get random seed on each edit use examples/workflow_Random.json  and add Load Image node.<br />
![example](https://github.com/myonmu0/ComfyUI-NanoGPT_ImageGeneration/blob/main/examples/2.png)


