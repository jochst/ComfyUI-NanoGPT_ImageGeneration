import requests
import json
import base64
import io
from PIL import Image, ImageOps
import numpy as np
import torch
import cv2
import torchvision.transforms as transforms

API_KEY = ""
URL = "https://nano-gpt.com/api/v1/images/generations"

IMAGE_MODELS = ["qwen-image", "z-image-turbo", "chroma", "hidream"]
IMAGE_SIZE = ["1024x1024", "1168x880", "880x1168", "1312x1312", "1312x1024", "1024x1312"]



def PngToTensors(b):
    image = Image.open(io.BytesIO(b))
    width,height = image.size
    image = ImageOps.exif_transpose(image)
    if image.mode == 'I':
        image = image.point(lambda i: i * (1 / 255))
    image = image.convert("RGB")

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
    ])

    # Apply transformations
    image_tensor = transform(image)
    image_tensor = image_tensor.permute(1, 2, 0).unsqueeze(0)

    return image_tensor[None,]


# tensors -> PNG image
def tensorsToB64Png(tensors):
    for (batch_number, image) in enumerate(tensors):
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        buff = io.BytesIO()
        img.save(buff, "PNG")
        buff.seek(0)
        base64_data = base64.b64encode(buff.read()).decode('utf-8')
        return base64_data

class GenerateImage:
    @classmethod
    def IS_CHANGED(s, prompt, model, size, step, strength, seed, image=None):
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
                "required": {
                    "prompt": ("STRING", {"multiline": True}),
                    "model": ( IMAGE_MODELS, {"default": "qwen-image"}),
                    "size": ( IMAGE_SIZE, {"default": "1024x1024"}),
                    "step": ("INT", {"default": 30}),
                    "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                    "seed": ("INT", {"default": 43}),
                  },
                "optional": {
                    "image": ("IMAGE", ),
                  }
                }

    RETURN_NAMES = ("image", )
    RETURN_TYPES = ("IMAGE", )
    CATEGORY = "NanoGPT"
    FUNCTION = "f"

    def f(self, prompt, model, size, step, strength, seed, image=None):
      headers = {
          "Authorization": "Bearer " + API_KEY,
          "Content-Type": "application/json"
      }
      json_data = {
          "model": model,
          "prompt": prompt,
          "size": size,
          "strength": strength,
          "seed": seed,
          "num_inference_steps": step,
          "response_format": "b64_json",
      }

      # Append reference image
      if image is not None:
        b64_img = tensorsToB64Png(image.contiguous())
        img_size = len(b64_img) / (1024 * 1024)
        #print(f"Image size: {img_size:4f}MB")
        if(img_size > 4):
          raise Exception('Reference image should be smaller than 4MB.')
        json_data["imageDataUrl"] = "data:image/jpeg;base64," + b64_img

      r = requests.post(URL, headers=headers, json=json_data)

      r.raise_for_status()
      r = str(r.json()).replace("\'", "\"")

      r = json.loads(str(r))
      r = r['data'][0]['b64_json']

      bImage = base64.b64decode(r)

      t = PngToTensors(bImage)
      return (t)   
  


NODE_CLASS_MAPPINGS = {
    "NanoGPT - Image Generation": GenerateImage,
}


NODE_DISPLAY_NAME_MAPPINGS = {
}
